/**
 * Expo config plugin that injects a VisionCamera frame processor plugin
 * for hand gesture detection using Apple's Vision framework.
 *
 * This adds two native files to the iOS project:
 * - HandGesturePlugin.swift — runs VNDetectHumanHandPoseRequest on each frame
 * - HandGesturePlugin.m — ObjC bridge to register the plugin with VisionCamera
 */
const {
  withXcodeProject,
  withDangerousMod,
} = require("@expo/config-plugins");
const fs = require("fs");
const path = require("path");

// ──────────────────────────────────────────────────────────────────────
// Native Swift code: VisionCamera frame processor plugin for hand pose
// ──────────────────────────────────────────────────────────────────────
const SWIFT_CODE = `
import Vision
import VisionCamera
import CoreMedia
import CoreVideo

@objc(DetectHandGesturePlugin)
public class DetectHandGesturePlugin: FrameProcessorPlugin {

    private lazy var handPoseRequest: VNDetectHumanHandPoseRequest = {
        let request = VNDetectHumanHandPoseRequest()
        request.maximumHandCount = 2
        return request
    }()

    public override func callback(
        _ frame: Frame,
        withArguments arguments: [AnyHashable: Any]?
    ) -> Any? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(frame.buffer) else {
            return nil
        }

        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: .up,
            options: [:]
        )

        do {
            try handler.perform([handPoseRequest])
        } catch {
            return nil
        }

        guard let observations = handPoseRequest.results, !observations.isEmpty else {
            return nil
        }

        var hands: [[String: Any]] = []

        for observation in observations {
            var hand: [String: Any] = [:]
            hand["chirality"] = observation.chirality == .left ? "left" : "right"

            let landmarks: [(String, VNHumanHandPoseObservation.JointName)] = [
                ("thumbTip",   .thumbTip),
                ("thumbIP",    .thumbIP),
                ("thumbMP",    .thumbMP),
                ("thumbCMC",   .thumbCMC),
                ("indexTip",   .indexTip),
                ("indexDIP",   .indexDIP),
                ("indexPIP",   .indexPIP),
                ("indexMCP",   .indexMCP),
                ("middleTip",  .middleTip),
                ("middleDIP",  .middleDIP),
                ("middlePIP",  .middlePIP),
                ("middleMCP",  .middleMCP),
                ("ringTip",    .ringTip),
                ("ringDIP",    .ringDIP),
                ("ringPIP",    .ringPIP),
                ("ringMCP",    .ringMCP),
                ("littleTip",  .littleTip),
                ("littleDIP",  .littleDIP),
                ("littlePIP",  .littlePIP),
                ("littleMCP",  .littleMCP),
                ("wrist",      .wrist),
            ]

            var allValid = true
            for (name, joint) in landmarks {
                if let point = try? observation.recognizedPoint(joint),
                   point.confidence > 0.2 {
                    hand[name] = [
                        Double(point.location.x),
                        Double(point.location.y),
                        Double(point.confidence),
                    ]
                } else {
                    allValid = false
                    break
                }
            }

            if allValid {
                hands.append(hand)
            }
        }

        if hands.isEmpty { return nil }
        return ["hands": hands] as [String: Any]
    }
}
`;

// ──────────────────────────────────────────────────────────────────────
// ObjC bridge: registers the Swift plugin with VisionCamera
// ──────────────────────────────────────────────────────────────────────
// NOTE: The ObjC file must import the generated Swift header so the macro
// can see the Swift class.  The header name matches the Xcode *module* name,
// which for Expo-generated projects equals the target name with hyphens
// replaced by underscores.  We resolve it at write-time from projectName.
const OBJC_TEMPLATE = (moduleName) => `
#import <VisionCamera/FrameProcessorPlugin.h>
#import <VisionCamera/FrameProcessorPluginRegistry.h>
#import "${moduleName}-Swift.h"

VISION_EXPORT_SWIFT_FRAME_PROCESSOR(DetectHandGesturePlugin, detectHandGesture)
`;

function withHandDetectionFiles(config) {
  return withDangerousMod(config, [
    "ios",
    (cfg) => {
      const projectRoot = cfg.modRequest.platformProjectRoot;
      const projectName = cfg.modRequest.projectName;
      const targetDir = path.join(projectRoot, projectName);

      fs.mkdirSync(targetDir, { recursive: true });
      fs.writeFileSync(
        path.join(targetDir, "HandGesturePlugin.swift"),
        SWIFT_CODE.trimStart()
      );
      // Derive the Swift module name (Xcode replaces non-alnum with underscores)
      const moduleName = projectName.replace(/[^a-zA-Z0-9]/g, "_");
      fs.writeFileSync(
        path.join(targetDir, "HandGesturePlugin.m"),
        OBJC_TEMPLATE(moduleName).trimStart()
      );
      return cfg;
    },
  ]);
}

function withHandDetectionXcode(config) {
  return withXcodeProject(config, (cfg) => {
    const project = cfg.modResults;
    const projectName = cfg.modRequest.projectName;

    // Find the main group for the app target
    const mainGroup = project.getFirstProject().firstProject.mainGroup;

    // Add source files — pass the group key so xcode lib can locate it
    project.addSourceFile(
      `${projectName}/HandGesturePlugin.swift`,
      { target: project.getFirstTarget().uuid },
      mainGroup
    );
    project.addSourceFile(
      `${projectName}/HandGesturePlugin.m`,
      { target: project.getFirstTarget().uuid },
      mainGroup
    );

    // Link Vision.framework
    project.addFramework("Vision.framework", { weak: false });

    return cfg;
  });
}

module.exports = function withHandDetection(config) {
  config = withHandDetectionFiles(config);
  config = withHandDetectionXcode(config);
  return config;
};
