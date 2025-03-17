/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
        "./src/components/ui/**/*.{js,ts,jsx,tsx}", // ✅ Include ShadCN UI components
    ],
    darkMode: "class",
    theme: {
        extend: {},
    },
    plugins: [],
};
