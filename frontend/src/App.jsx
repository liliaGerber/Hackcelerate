import {useEffect, useState} from "react";
import RadarChart from "./components/Graph/RadarChart.jsx";

function App() {
    const [data, setData] = useState({
        EXT: 0,
        NEU: 0,
        AGR: 0,
        OPN: 0,
        CON: 0,
        name: ''
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
                name: newData[5],
            });
        };

        return () => {
            // socket.close();
        };
    }, []);

    return <div>
        <h2>{data.name}</h2>
        <RadarChart data={{
            EXT: data.EXT,
            NEU: data.NEU,
            AGR: data.AGR,
            OPN: data.OPN,
            CON: data.CON,
        }}/>
    </div>;
}

export default App;
