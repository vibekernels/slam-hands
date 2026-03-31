import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  Pressable,
  Alert,
  ActivityIndicator,
} from "react-native";
import { getServerUrl, setServerUrl } from "@/lib/storage";
import { checkServerConnection } from "@/lib/upload";

export default function SettingsScreen() {
  const [url, setUrl] = useState("");
  const [testing, setTesting] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    getServerUrl().then(setUrl);
  }, []);

  const handleSave = async () => {
    await setServerUrl(url.trim());
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handleTest = async () => {
    setTesting(true);
    await setServerUrl(url.trim());
    const ok = await checkServerConnection();
    setTesting(false);
    Alert.alert(
      ok ? "Connected" : "Failed",
      ok
        ? "Server is reachable and responding."
        : "Could not reach server. Check the URL and ensure the server is running."
    );
  };

  return (
    <View style={styles.container}>
      <Text style={styles.label}>Annotation Server URL</Text>
      <TextInput
        style={styles.input}
        value={url}
        onChangeText={setUrl}
        placeholder="http://192.168.1.100:8000"
        placeholderTextColor="#666"
        autoCapitalize="none"
        autoCorrect={false}
        keyboardType="url"
      />

      <View style={styles.row}>
        <Pressable style={styles.button} onPress={handleSave}>
          <Text style={styles.buttonText}>
            {saved ? "Saved!" : "Save"}
          </Text>
        </Pressable>

        <Pressable
          style={[styles.button, styles.testButton]}
          onPress={handleTest}
          disabled={testing}
        >
          {testing ? (
            <ActivityIndicator color="#fff" size="small" />
          ) : (
            <Text style={styles.buttonText}>Test Connection</Text>
          )}
        </Pressable>
      </View>

      <Text style={styles.hint}>
        The server should be running visualizer.py in upload mode.{"\n"}
        Uploaded clips will be annotated with SLAM, hand pose, and speech
        transcription automatically.
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#000",
    padding: 24,
  },
  label: {
    color: "#aaa",
    fontSize: 14,
    marginBottom: 8,
    marginTop: 16,
  },
  input: {
    backgroundColor: "#1a1a1a",
    color: "#fff",
    fontSize: 16,
    padding: 14,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#333",
  },
  row: {
    flexDirection: "row",
    gap: 12,
    marginTop: 16,
  },
  button: {
    backgroundColor: "#ff3b30",
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 10,
    alignItems: "center",
    justifyContent: "center",
    minWidth: 100,
  },
  testButton: {
    backgroundColor: "#333",
    flex: 1,
  },
  buttonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },
  hint: {
    color: "#666",
    fontSize: 13,
    marginTop: 24,
    lineHeight: 20,
  },
});
