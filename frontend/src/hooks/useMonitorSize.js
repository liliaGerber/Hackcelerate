import { useState, useEffect } from "react";

export const useMonitorSize = () => {
    const [browserWidth, setBrowserWidth] = useState(window.innerWidth);
    const [deviceWidth, setDeviceWidth] = useState(screen.width);
    const [isMobile, setIsMobile] = useState(window.innerWidth <= 650);

    useEffect(() => {
        const handleResize = () => {
            setBrowserWidth(window.innerWidth);
            setDeviceWidth(screen.width);
            setIsMobile(window.innerWidth <= 650);
        };

        window.addEventListener("resize", handleResize);

        // Cleanup listener on unmount
        return () => {
            window.removeEventListener("resize", handleResize);
        };
    }, []);

    return { browserWidth, deviceWidth, isMobile };
};
