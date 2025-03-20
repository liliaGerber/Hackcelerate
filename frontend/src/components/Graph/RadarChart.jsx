import Chart from "react-apexcharts";
import PropTypes from "prop-types";
import { useEffect, useState } from "react";
import './Chart.css';

const RadarChart = ({ data }) => {
    const [options, setOptions] = useState({});
    const [series, setSeries] = useState([]);

    useEffect(() => {
        let keys = [];
        let values = [];

        for (let key in data) {
            keys.push(key.toUpperCase());
            values.push(data[key]);
        }

        setSeries([
            {
                name: "Audio Feature",
                data: values,
            },
            {
                name: "Option A",
                data: values.map(() => 1.0),
            },
        ]);
        setOptions({
            plotOptions: {
                radar: {
                    polygons: {
                        fill: {
                            colors: ["#0F0F0F", "#363636"]
                        }
                    }
                }
            },
            chart: {
                toolbar: {
                    show: false,
                },
            },
            legend: {
                show: false,
            },
            yaxis: {
                show: false,
            },
            xaxis: {
                categories: keys,
                labels: {
                    show: true,
                    style: {
                        colors: ["#F1F1F1"],
                        fontSize: "12px",
                    }
                },
            },
            fill: {
                colors: ["#24D34E", "#000000"],
                opacity: [0.7, 0.0],
            },
            stroke: {
                show: false
            },
            markers: {
                size: [3, 0],
                colors: ["#1A8833", "#000000"],
                hover: {
                    size: 6
                }
            },
            grid: {
                show: false,
            },
            tooltip: {
                enabled: true,
                theme: "dark",
            }
        });
    }, [data]);

    return (
        <div>
            <Chart options={options}
                   series={series}
                   type="radar"
                   height={350}
                   width={350}/>
        </div>
    );
};

RadarChart.propTypes = {
    data: PropTypes.shape({
        EXT: PropTypes.number.isRequired,
        NEU: PropTypes.number.isRequired,
        AGR: PropTypes.number.isRequired,
        CON: PropTypes.number.isRequired,
        OPN: PropTypes.number.isRequired,
    }).isRequired,
};

export default RadarChart;