import {useEffect, useState} from "react";
import RadarChart from "./components/Graph/RadarChart.jsx";

function App() {
    const [data, setData] = useState({
        EXT: 0,
        NEU: 0,
        AGR: 0,
        CON: 0,
        OPN: 0,
        name: '',
        text: ''
    });

    useEffect(() => {
        const socket = new WebSocket("ws://localhost:5000/ws/personality-visualizer"); // Adjust to your backend URL

        socket.onmessage = (event) => {
            console.log("Received message", event.data);
            const newData = JSON.parse(event.data);
            setData({
                EXT: newData[0],
                NEU: newData[1],
                AGR: newData[2],
                CON: newData[4],
                OPN: newData[3],
                name: newData[5],
                text: newData[6]
            });
        };

        return () => {
            // socket.close();
        };
    }, []);

    return <div>
        <h2>{data.name}</h2>
        <h4>{data.text}</h4>
        <RadarChart data={{
            EXT: data.EXT,
            NEU: data.NEU,
            AGR: data.AGR,
            CON: data.CON,
            OPN: data.OPN,
        }}/>
    </div>;
}

export default App;
