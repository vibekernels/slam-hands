/**
 * Persistent storage helpers using AsyncStorage and expo-file-system.
 */
import AsyncStorage from "@react-native-async-storage/async-storage";
import { File, Directory, Paths } from "expo-file-system";

const SERVER_URL_KEY = "ego_server_url";
const DEFAULT_SERVER_URL = "http://192.168.1.100:8000";

// ── Server URL ──────────────────────────────────────────────────────

export async function getServerUrl(): Promise<string> {
  const url = await AsyncStorage.getItem(SERVER_URL_KEY);
  return url || DEFAULT_SERVER_URL;
}

export async function setServerUrl(url: string): Promise<void> {
  await AsyncStorage.setItem(SERVER_URL_KEY, url);
}

// ── Clips directory ─────────────────────────────────────────────────

function getClipsDir(): Directory {
  const dir = new Directory(Paths.document, "clips");
  if (!dir.exists) {
    dir.create();
  }
  return dir;
}

export interface ClipInfo {
  uri: string;
  filename: string;
  createdAt: number;
  duration?: number;
  uploaded: boolean;
  uploading: boolean;
  error?: string;
}

const CLIPS_META_KEY = "ego_clips";

export async function getClips(): Promise<ClipInfo[]> {
  const raw = await AsyncStorage.getItem(CLIPS_META_KEY);
  if (!raw) return [];
  try {
    return JSON.parse(raw);
  } catch {
    return [];
  }
}

export async function saveClipMeta(clip: ClipInfo): Promise<void> {
  const clips = await getClips();
  clips.unshift(clip);
  await AsyncStorage.setItem(CLIPS_META_KEY, JSON.stringify(clips));
}

export async function updateClipMeta(
  filename: string,
  update: Partial<ClipInfo>
): Promise<void> {
  const clips = await getClips();
  const idx = clips.findIndex((c) => c.filename === filename);
  if (idx >= 0) {
    clips[idx] = { ...clips[idx], ...update };
    await AsyncStorage.setItem(CLIPS_META_KEY, JSON.stringify(clips));
  }
}

export async function deleteClipMeta(filename: string): Promise<void> {
  const clips = await getClips();
  const filtered = clips.filter((c) => c.filename !== filename);
  await AsyncStorage.setItem(CLIPS_META_KEY, JSON.stringify(filtered));
}

/**
 * Save a recorded video to the clips directory and record its metadata.
 * Returns the new ClipInfo.
 */
export async function saveClip(
  sourceUri: string,
  duration?: number
): Promise<ClipInfo> {
  const dir = getClipsDir();
  const timestamp = Date.now();
  const filename = `ego_${timestamp}.mov`;

  const sourceFile = new File(sourceUri);
  const destFile = new File(dir, filename);
  sourceFile.move(destFile);

  const clip: ClipInfo = {
    uri: destFile.uri,
    filename,
    createdAt: timestamp,
    duration,
    uploaded: false,
    uploading: false,
  };

  await saveClipMeta(clip);
  return clip;
}

/**
 * Delete a clip file from disk.
 */
export function deleteClipFile(uri: string): void {
  try {
    const file = new File(uri);
    if (file.exists) {
      file.delete();
    }
  } catch {}
}
