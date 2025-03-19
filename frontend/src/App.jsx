import {Suspense} from 'react';
import '@fontsource/roboto/300.css';
import '@fontsource/roboto/400.css';
import '@fontsource/roboto/500.css';
import '@fontsource/roboto/700.css';

import {BrowserRouter as Router, useRoutes} from "react-router-dom";
import routes from "./routes.jsx";
import Navbar from "./components/Navbar";
import {createTheme, ThemeProvider} from "@mui/material";
import colourTheme from "@/settings/colourTheme.json";

const AppRoutes = () => {
    return useRoutes(routes);
};

function App() {

    const theme = createTheme({
        palette: {
            primary: {
                main: colourTheme.primary,
            },
            secondary: {
                main: colourTheme.secondary,
            },
            info: {
                main: colourTheme.info,
            },
            warning: {
                main: colourTheme.warning,
            },
            success: {
                main: colourTheme.success,
            },
            error: {
                main: colourTheme.error,
            },
        },
    });

    return (
        <Router>
            <ThemeProvider theme={theme}>
            <Navbar/>
            <Suspense fallback={<div>Loading...</div>}>
                <AppRoutes/>
            </Suspense>
            </ThemeProvider>
        </Router>
    );
}

export default App;

