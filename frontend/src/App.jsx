import { useEffect, useState } from "react";
import RadarChart from "./components/Graph/RadarChart.jsx";

function App() {
    const [data, setData] = useState({
        EXT: 2,
        NEU: 2,
        AGR: 1,
        OPN: 3,
        CON: 4,
    });

    useEffect(() => {
        const socket = new WebSocket("ws://localhost:5000/ws"); // Adjust to your backend URL

        socket.onmessage = (event) => {
            console.log("Received message", event.data);
            const newData = JSON.parse(event.data);
            setData({
                EXT: newData[0],
                NEU: newData[1],
                AGR: newData[2],
                OPN: newData[3],
                CON: newData[4],
            });
        };

        return () => {
            // socket.close();
        };
    }, []);

    return <RadarChart data={data} />;
}

export default App;
