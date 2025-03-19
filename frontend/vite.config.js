import {defineConfig} from 'vite'
import react from '@vitejs/plugin-react'
import path from "path";
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
    plugins: [react(), tailwindcss()],
    resolve: {
        alias: {
            "@": path.resolve(__dirname, "src"),
        },
    },
    server: {
        proxy: {
            "/api": {
                target: process.env.VITE_API_ENDPOINT,
                changeOrigin: true,
                secure: false,
            },
        },
        host: "0.0.0.0",
        port: 5173,
        strictPort: true,
    },
})
