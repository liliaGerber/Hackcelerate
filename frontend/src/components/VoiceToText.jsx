import React, { useEffect } from "react";
import { Mic, MicOff } from "@mui/icons-material";
import { useVoiceToText } from "react-speakup";
import { Button } from "@mui/material";
import useMicrophonePermission from "@/hooks/useMicrophonePermission"; // Import permission hook

const VoiceToText = () => {
    const { permission, requestPermission, error } = useMicrophonePermission();

    // Fetching the speech recognition values from react-speakup
    const { startListening, stopListening, transcript, isListening, error: speechError } = useVoiceToText({
        continuous: true, // Keeps listening until stopped
        language: "en-US", // Set language
        interimResults: true, // Shows words as they're being spoken
        debug: true, // Logs internal debugging messages
    });

    // 🔹 Debugging: Log transcript updates
    useEffect(() => {
        console.log("📝 Transcript updated:", transcript);
    }, [transcript]);

    // 🔹 Debugging: Log listening state changes
    useEffect(() => {
        console.log(`🎙️ Listening status changed: ${isListening}`);
    }, [isListening]);

    // 🔹 Debugging: Log errors
    useEffect(() => {
        if (speechError) {
            console.error("❌ Speech recognition error:", speechError);
        }
    }, [speechError]);

    const handleStartListening = () => {
        if (typeof startListening !== "function") {
            console.error("🚨 startListening is not a function! Check the library setup.");
            return;
        }
        if (permission === "granted") {
            console.log("🎙️ Starting voice recognition...");
            startListening();
        } else {
            console.warn("⚠️ Microphone permission is not granted. Requesting permission...");
            requestPermission();
        }
    };

    const handleStopListening = () => {
        if (typeof stopListening !== "function") {
            console.error("🚨 stopListening is not a function! Check the library setup.");
            return;
        }
        console.log("🛑 Stopping voice recognition...");
        stopListening();
    };

    return (
        <div className="flex flex-col gap-6">
            <div className="flex gap-6">
                <Button onClick={handleStartListening} className="bg-primary">
                    <Mic /> {isListening ? "Listening..." : "Start"}
                </Button>
                <Button onClick={handleStopListening}>
                    <MicOff /> Stop
                </Button>
            </div>

            <h2>Transcript: {transcript || "🗣️ No speech detected yet."}</h2>

            {permission !== "granted" && (
                <p className="text-red-500">
                    ⚠️ Microphone permission is <strong>{permission}</strong>. Click "Start" to request access.
                </p>
            )}
            {error && <p className="text-red-500">🚨 Permission Error: {error}</p>}
            {speechError && <p className="text-red-500">❌ Speech Recognition Error: {speechError}</p>}
        </div>
    );
};

export default VoiceToText;
