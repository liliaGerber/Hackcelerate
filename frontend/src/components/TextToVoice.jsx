import React from "react";
import { useTextToVoice } from "react-speakup";
import { Play, Pause, StepForward } from "lucide-react";

const TextToVoice = () => {
    const { speak, isSpeaking, pause, ref, resume, setVoice, voices } =
    useTextToVoice<HTMLDivElement>({ pitch: 1, rate: 1, volume: 1 });

    return (
        <div ref={ref} className="flex flex-col items-center ">
            <div className="flex justify-between w-full">
                <h1 className="font-bold text-xl mb-4">speech synthesis</h1>
                <select
                    onChange={(event) =>
                        setVoice(event.target.value)
                    }
                >
                    {voices.map((voice) => (
                        <option key={voice}>{voice}</option>
                    ))}
                </select>
            </div>
            <p className="w-[800px] mb-10">
                The SpeechSynthesis interface of the Web Speech API serves as the
                controller interface for the speech service. It allows you to retrieve
                information about the available synthesis voices on the device, initiate
                and pause speech, and execute other related commands.
            </p>
            <div className="flex gap-6">
                <Play onClick={speak} role="button" />
                <Pause onClick={pause} role="button" />
                <StepForward onClick={resume} role="button" />
            </div>
            {isSpeaking && "is speaking"}
        </div>
    );
};

export default TextToVoice;