import React, { useCallback, useState } from "react";
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  Pressable,
  Alert,
  ActivityIndicator,
  RefreshControl,
} from "react-native";
import { useFocusEffect, useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import {
  getClips,
  deleteClipMeta,
  deleteClipFile,
  updateClipMeta,
  type ClipInfo,
} from "@/lib/storage";
import { uploadClip } from "@/lib/upload";

export default function ClipsScreen() {
  const [clips, setClips] = useState<ClipInfo[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const router = useRouter();

  const loadClips = useCallback(async () => {
    setClips(await getClips());
  }, []);

  useFocusEffect(
    useCallback(() => {
      loadClips();
    }, [loadClips])
  );

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadClips();
    setRefreshing(false);
  };

  const handleUpload = async (clip: ClipInfo) => {
    await updateClipMeta(clip.filename, { uploading: true, error: undefined });
    await loadClips();

    const result = await uploadClip(clip.uri, clip.filename);

    await updateClipMeta(clip.filename, {
      uploading: false,
      uploaded: result.success,
      error: result.error,
    });
    await loadClips();

    if (!result.success) {
      Alert.alert("Upload Failed", result.error);
    }
  };

  const handleDelete = (clip: ClipInfo) => {
    Alert.alert("Delete Clip", `Delete ${clip.filename}?`, [
      { text: "Cancel", style: "cancel" },
      {
        text: "Delete",
        style: "destructive",
        onPress: async () => {
          deleteClipFile(clip.uri);
          await deleteClipMeta(clip.filename);
          await loadClips();
        },
      },
    ]);
  };

  const renderClip = ({ item }: { item: ClipInfo }) => {
    const date = new Date(item.createdAt);
    const timeStr = date.toLocaleTimeString();
    const dateStr = date.toLocaleDateString();

    return (
      <View style={styles.clipRow}>
        <View style={styles.clipInfo}>
          <Text style={styles.clipName}>{item.filename}</Text>
          <Text style={styles.clipDate}>
            {dateStr} {timeStr}
            {item.duration ? ` | ${item.duration.toFixed(1)}s` : ""}
          </Text>
          {item.error && <Text style={styles.clipError}>{item.error}</Text>}
        </View>

        <View style={styles.clipActions}>
          {item.uploading ? (
            <ActivityIndicator color="#ff3b30" />
          ) : item.uploaded ? (
            <Ionicons name="checkmark-circle" size={28} color="#34c759" />
          ) : (
            <Pressable onPress={() => handleUpload(item)}>
              <Ionicons name="cloud-upload" size={28} color="#007aff" />
            </Pressable>
          )}

          <Pressable onPress={() => handleDelete(item)} style={{ marginLeft: 16 }}>
            <Ionicons name="trash" size={24} color="#666" />
          </Pressable>
        </View>
      </View>
    );
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Recorded Clips</Text>
        <Pressable onPress={() => router.push("/settings")}>
          <Ionicons name="settings-outline" size={24} color="#aaa" />
        </Pressable>
      </View>

      <FlatList
        data={clips}
        keyExtractor={(item) => item.filename}
        renderItem={renderClip}
        contentContainerStyle={clips.length === 0 ? styles.empty : undefined}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={handleRefresh}
            tintColor="#666"
          />
        }
        ListEmptyComponent={
          <View style={styles.emptyContent}>
            <Ionicons name="film-outline" size={64} color="#333" />
            <Text style={styles.emptyText}>No clips yet</Text>
            <Text style={styles.emptyHint}>
              Make the OK sign with both hands to start recording
            </Text>
          </View>
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#000",
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    padding: 16,
    paddingTop: 60,
    borderBottomWidth: 1,
    borderBottomColor: "#222",
  },
  title: {
    color: "#fff",
    fontSize: 22,
    fontWeight: "700",
  },
  clipRow: {
    flexDirection: "row",
    alignItems: "center",
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: "#1a1a1a",
  },
  clipInfo: {
    flex: 1,
  },
  clipName: {
    color: "#fff",
    fontSize: 14,
    fontFamily: "monospace",
  },
  clipDate: {
    color: "#888",
    fontSize: 12,
    marginTop: 4,
  },
  clipError: {
    color: "#ff3b30",
    fontSize: 12,
    marginTop: 4,
  },
  clipActions: {
    flexDirection: "row",
    alignItems: "center",
  },
  empty: {
    flex: 1,
  },
  emptyContent: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingTop: 120,
  },
  emptyText: {
    color: "#666",
    fontSize: 18,
    marginTop: 16,
  },
  emptyHint: {
    color: "#444",
    fontSize: 14,
    marginTop: 8,
    textAlign: "center",
    paddingHorizontal: 40,
  },
});
