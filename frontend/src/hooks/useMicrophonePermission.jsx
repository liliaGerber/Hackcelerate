import { useState, useEffect } from "react";

const useMicrophonePermission = () => {
    const [permission, setPermission] = useState("unknown"); // "granted" | "denied" | "prompt" | "unknown"
    const [error, setError] = useState(null);

    useEffect(() => {
        if (typeof navigator === "undefined" || !navigator.permissions) {
            console.warn("🚨 Navigator is not available. Ensure this is running in a browser.");
            return;
        }

        console.log("🔍 Checking microphone permission...");
        navigator.permissions.query({ name: "microphone" })
            .then(result => {
                console.log(`✅ Microphone permission status: ${result.state}`);
                setPermission(result.state);

                // Listen for permission changes
                result.onchange = () => {
                    console.log(`🔄 Microphone permission changed: ${result.state}`);
                    setPermission(result.state);
                };
            })
            .catch(err => {
                console.error("❌ Error checking microphone permission:", err.message);
                setError(err.message);
            });
    }, []);

    const requestPermission = async () => {
        try {
            console.log("🔹 Requesting microphone access...");
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            console.log("🎤 Microphone access granted!");
            setPermission("granted");

            // Release the microphone after permission is granted
            stream.getTracks().forEach(track => track.stop());
        } catch (err) {
            console.error("🚫 Microphone access denied:", err.message);
            setPermission("denied");
            setError(err.message);
        }
    };

    return { permission, requestPermission, error };
};

export default useMicrophonePermission;
