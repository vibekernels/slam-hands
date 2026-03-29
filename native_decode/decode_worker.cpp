/*
 * AsyncVideoDecoder: GIL-free video decode + resize for SLAM pipeline.
 *
 * Runs FFmpeg decode + swscale resize in a native C++ thread that NEVER
 * acquires the Python GIL. The Python main thread reads results via get_next()
 * and can simultaneously run SLAM.track() (which executes CUDA kernels that
 * also release the GIL).
 *
 * Two modes:
 *   slam_only=true:  Only produce small BGR SLAM tensors (fast, ~4s for 1858 frames)
 *   slam_only=false: Produce both full RGB + SLAM BGR (slower, ~10s)
 *
 * Build: python setup.py build_ext --inplace
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <vector>
#include <stdexcept>
#include <cstring>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

namespace py = pybind11;

struct DecodedFrame {
    std::vector<uint8_t> rgb_data;   // Full-res RGB (empty if slam_only)
    int rgb_w = 0, rgb_h = 0;

    std::vector<uint8_t> slam_data;  // Resized BGR for SLAM
    int slam_w = 0, slam_h = 0;
};

class AsyncVideoDecoder {
public:
    AsyncVideoDecoder() = default;
    ~AsyncVideoDecoder() { stop(); }

    void start(const std::string& video_path, int slam_w, int slam_h,
               bool slam_only = false, int queue_depth = 64) {
        stop();
        slam_w_ = slam_w;
        slam_h_ = slam_h;
        slam_only_ = slam_only;
        queue_depth_ = queue_depth;
        done_ = false;
        error_.clear();
        frame_count_ = 0;
        worker_ = std::thread(&AsyncVideoDecoder::decode_loop, this, video_path);
    }

    // Returns numpy pair or None. Releases GIL while waiting for next frame.
    py::object get_next() {
        DecodedFrame frame;
        bool got_frame = false;

        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]{ return !queue_.empty() || done_; });
            if (!queue_.empty()) {
                frame = std::move(queue_.front());
                queue_.pop();
                got_frame = true;
                cv_producer_.notify_one();
            }
        }

        if (!got_frame) {
            if (!error_.empty()) throw std::runtime_error(error_);
            return py::none();
        }

        // Build slam numpy array
        auto slam = py::array_t<uint8_t>({frame.slam_h, frame.slam_w, 3});
        std::memcpy(slam.mutable_data(), frame.slam_data.data(), frame.slam_data.size());

        if (slam_only_) {
            return py::make_tuple(py::none(), slam);
        }

        auto rgb = py::array_t<uint8_t>({frame.rgb_h, frame.rgb_w, 3});
        std::memcpy(rgb.mutable_data(), frame.rgb_data.data(), frame.rgb_data.size());
        return py::make_tuple(rgb, slam);
    }

    int frame_count() const { return frame_count_; }
    bool is_done() const { return done_; }

    void stop() {
        done_ = true;
        cv_producer_.notify_all();
        if (worker_.joinable()) worker_.join();
        std::lock_guard<std::mutex> lock(mutex_);
        while (!queue_.empty()) queue_.pop();
    }

    static py::tuple get_metadata(const std::string& video_path) {
        AVFormatContext* fmt_ctx = nullptr;
        if (avformat_open_input(&fmt_ctx, video_path.c_str(), nullptr, nullptr) < 0)
            throw std::runtime_error("Cannot open: " + video_path);
        avformat_find_stream_info(fmt_ctx, nullptr);

        int idx = -1;
        for (unsigned i = 0; i < fmt_ctx->nb_streams; i++)
            if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
                { idx = i; break; }
        if (idx < 0) { avformat_close_input(&fmt_ctx); throw std::runtime_error("No video"); }

        auto* s = fmt_ctx->streams[idx];
        auto result = py::make_tuple(
            av_q2d(s->avg_frame_rate), s->codecpar->width, s->codecpar->height, s->nb_frames);
        avformat_close_input(&fmt_ctx);
        return result;
    }

private:
    void process_frame(AVFrame* frame, int src_w, int src_h,
                       SwsContext* sws_rgb, AVFrame* rgb_frame,
                       SwsContext* sws_slam, uint8_t* slam_buf, int slam_linesize) {
        DecodedFrame df;
        df.slam_w = slam_w_;
        df.slam_h = slam_h_;

        // SLAM resize (always — small output, fast)
        uint8_t* slam_planes[1] = { slam_buf };
        int slam_strides[1] = { slam_linesize };
        sws_scale(sws_slam, frame->data, frame->linesize, 0, src_h,
                  slam_planes, slam_strides);
        df.slam_data.assign(slam_buf, slam_buf + slam_h_ * slam_linesize);

        // Full-res RGB (optional)
        if (!slam_only_) {
            df.rgb_w = src_w;
            df.rgb_h = src_h;
            sws_scale(sws_rgb, frame->data, frame->linesize, 0, src_h,
                      rgb_frame->data, rgb_frame->linesize);
            df.rgb_data.resize(src_h * src_w * 3);
            for (int y = 0; y < src_h; y++)
                std::memcpy(df.rgb_data.data() + y * src_w * 3,
                            rgb_frame->data[0] + y * rgb_frame->linesize[0], src_w * 3);
        }

        std::unique_lock<std::mutex> lock(mutex_);
        cv_producer_.wait(lock, [this]{ return (int)queue_.size() < queue_depth_ || done_; });
        if (!done_) {
            queue_.push(std::move(df));
            frame_count_++;
        }
        lock.unlock();
        cv_.notify_one();
    }

    void decode_loop(std::string video_path) {
        AVFormatContext* fmt_ctx = nullptr;
        AVCodecContext* dec_ctx = nullptr;
        SwsContext* sws_rgb = nullptr;
        SwsContext* sws_slam = nullptr;
        AVFrame* frame = nullptr;
        AVFrame* rgb_frame = nullptr;
        AVPacket* pkt = nullptr;

        try {
            if (avformat_open_input(&fmt_ctx, video_path.c_str(), nullptr, nullptr) < 0)
                throw std::runtime_error("Cannot open video");
            avformat_find_stream_info(fmt_ctx, nullptr);

            int video_idx = -1;
            for (unsigned i = 0; i < fmt_ctx->nb_streams; i++)
                if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
                    { video_idx = i; break; }
            if (video_idx < 0) throw std::runtime_error("No video stream");

            auto* codecpar = fmt_ctx->streams[video_idx]->codecpar;
            const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
            if (!codec) throw std::runtime_error("Codec not found");

            dec_ctx = avcodec_alloc_context3(codec);
            avcodec_parameters_to_context(dec_ctx, codecpar);
            dec_ctx->thread_count = 0;
            dec_ctx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
            if (avcodec_open2(dec_ctx, codec, nullptr) < 0)
                throw std::runtime_error("Cannot open codec");

            int src_w = dec_ctx->width, src_h = dec_ctx->height;

            // swscale: source → BGR24 SLAM resolution (always needed)
            sws_slam = sws_getContext(src_w, src_h, dec_ctx->pix_fmt,
                slam_w_, slam_h_, AV_PIX_FMT_BGR24,
                SWS_BILINEAR, nullptr, nullptr, nullptr);

            // swscale: source → RGB24 full resolution (optional)
            if (!slam_only_) {
                sws_rgb = sws_getContext(src_w, src_h, dec_ctx->pix_fmt,
                    src_w, src_h, AV_PIX_FMT_RGB24,
                    SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
            }

            frame = av_frame_alloc();
            pkt = av_packet_alloc();

            rgb_frame = av_frame_alloc();
            if (!slam_only_) {
                rgb_frame->format = AV_PIX_FMT_RGB24;
                rgb_frame->width = src_w;
                rgb_frame->height = src_h;
                av_frame_get_buffer(rgb_frame, 32);
            }

            int slam_linesize = slam_w_ * 3;
            std::vector<uint8_t> slam_buf(slam_h_ * slam_linesize);

            // Main decode loop
            while (av_read_frame(fmt_ctx, pkt) >= 0) {
                if (pkt->stream_index != video_idx) { av_packet_unref(pkt); continue; }
                avcodec_send_packet(dec_ctx, pkt);
                av_packet_unref(pkt);
                while (avcodec_receive_frame(dec_ctx, frame) >= 0) {
                    if (done_) goto cleanup;
                    process_frame(frame, src_w, src_h, sws_rgb, rgb_frame,
                                  sws_slam, slam_buf.data(), slam_linesize);
                }
            }

            // Flush
            avcodec_send_packet(dec_ctx, nullptr);
            while (avcodec_receive_frame(dec_ctx, frame) >= 0) {
                if (done_) goto cleanup;
                process_frame(frame, src_w, src_h, sws_rgb, rgb_frame,
                              sws_slam, slam_buf.data(), slam_linesize);
            }

        } catch (const std::exception& e) {
            error_ = e.what();
        }

cleanup:
        if (frame) av_frame_free(&frame);
        if (rgb_frame) av_frame_free(&rgb_frame);
        if (pkt) av_packet_free(&pkt);
        if (sws_rgb) sws_freeContext(sws_rgb);
        if (sws_slam) sws_freeContext(sws_slam);
        if (dec_ctx) avcodec_free_context(&dec_ctx);
        if (fmt_ctx) avformat_close_input(&fmt_ctx);

        done_ = true;
        cv_.notify_all();
    }

    int slam_w_ = 0, slam_h_ = 0;
    bool slam_only_ = false;
    int queue_depth_ = 64;
    std::atomic<bool> done_{false};
    std::atomic<int> frame_count_{0};
    std::string error_;
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable cv_, cv_producer_;
    std::queue<DecodedFrame> queue_;
};


PYBIND11_MODULE(native_decode, m) {
    m.doc() = "GIL-free async video decoder for SLAM pipeline";

    py::class_<AsyncVideoDecoder>(m, "AsyncVideoDecoder")
        .def(py::init<>())
        .def("start", &AsyncVideoDecoder::start,
             py::arg("video_path"), py::arg("slam_w"), py::arg("slam_h"),
             py::arg("slam_only") = false, py::arg("queue_depth") = 64,
             "Start async decode. slam_only=True skips full-res RGB (2x faster).")
        .def("get_next", &AsyncVideoDecoder::get_next,
             "Get (rgb_or_None, slam_bgr) numpy pair. Returns None when done. "
             "Releases GIL while waiting.")
        .def("stop", &AsyncVideoDecoder::stop)
        .def("frame_count", &AsyncVideoDecoder::frame_count)
        .def("is_done", &AsyncVideoDecoder::is_done)
        .def_static("get_metadata", &AsyncVideoDecoder::get_metadata);
}
