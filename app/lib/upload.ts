/**
 * Upload service for sending recorded video clips to the backend.
 * The backend expects: POST /upload?filename=<name> with raw video bytes.
 */
import { File } from "expo-file-system";
import { getServerUrl } from "./storage";

export interface UploadResult {
  success: boolean;
  error?: string;
}

/**
 * Upload a video file to the annotation server.
 * Reads the file as a Blob and sends it via fetch.
 */
export async function uploadClip(
  fileUri: string,
  filename: string
): Promise<UploadResult> {
  const serverUrl = await getServerUrl();
  if (!serverUrl) {
    return { success: false, error: "No server URL configured" };
  }

  const url = `${serverUrl}/upload?filename=${encodeURIComponent(filename)}`;

  try {
    // expo-file-system v19 File implements Blob interface
    const file = new File(fileUri);
    const blob = await file.arrayBuffer();

    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/octet-stream",
      },
      body: blob,
    });

    if (res.ok) {
      return { success: true };
    }
    const text = await res.text();
    return {
      success: false,
      error: `Server returned ${res.status}: ${text}`,
    };
  } catch (err: any) {
    return {
      success: false,
      error: err.message || "Upload failed",
    };
  }
}

/**
 * Check if the server is reachable.
 */
export async function checkServerConnection(): Promise<boolean> {
  const serverUrl = await getServerUrl();
  if (!serverUrl) return false;

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000);
    const res = await fetch(`${serverUrl}/`, { signal: controller.signal });
    clearTimeout(timeout);
    return res.ok;
  } catch {
    return false;
  }
}
