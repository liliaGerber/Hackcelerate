/** @type {import('tailwindcss').Config} */
import theme from "@/settings/colourTheme.json"
import tailwindExtensioncolours from "@/settings/tailwindExtendColours.json"
export default{
    purge: [
        './src/**/*.html',
        './src/**/*.jsx',
    ],// remove unused styles in production
    darkMode: "media", // or 'media' or 'class'
    theme: {
        colors: theme,
        extend: {
            screens: {
                sm: '480px',
                md: '768px',
                lg: '976px',
                xl: '1440px',
            },
            colors: tailwindExtensioncolours,
            fontFamily: {
                sans: ['Graphik', 'sans-serif'],
                serif: ['Merriweather', 'serif'],
            },
            spacing: {
                '128': '32rem',
                '144': '36rem',
            },
            borderRadius: {
                '4xl': '2rem',
            }
        },
        plugins: [],
    }
}

