import React, { useEffect, useState } from "react";
import { Container, Typography, CircularProgress, Box } from "@mui/material";
import axios from "axios";
import {fetchHello} from "@/api/generalApi.js";
import VoiceToText from "@/components/VoiceToText.jsx";

const HomeScreen = () => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

/*    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await fetchHello();
                setData(response)
                return false;

            } catch (err) {
                setError("Failed to load data");
            } finally {
                setLoading(false);
            }
        };

        fetchData().then(response => setLoading(response));
    }, []);*/

    return (
        <Container maxWidth={false} className="min-h-screen flex justify-center items-center bg-primary-main px-4">
            <Box className="w-full max-w-2xl flex flex-col items-center text-center">
                <Typography variant="h2" className="text-primary font-bold mb-4">
                    Home Screen
                </Typography>
 {/*               {loading ? (
                    <CircularProgress color="primary" />
                ) : error ? (
                    <Typography variant="body1" className="text-red-500">
                        {error}
                    </Typography>
                ) : (
                    <Typography variant="body1" className="text-white">
                        {JSON.stringify(data, null, 2)}
                    </Typography>
                )}*/}
                <VoiceToText></VoiceToText>
            </Box>
        </Container>
    );
};

export default HomeScreen;
