// src/routes.tsx
import { lazy, Suspense } from "react";
// Lazy-loaded pages for better performance
const Home = lazy(() => import("./screens/HomeScreen"));
const Team = lazy(() => import("./screens/TeamScreen"));
const Login = lazy(() => import("./screens/LoginScreen"));
const Contact = lazy(() => import("./screens/ContactScreen"));

// { path: "*", element: <NotFound /> }, // 404 page

// Define all app routes in a centralized array
const routes = [
    { path: "/", element: <Home /> },
    { path: "/team", element: <Team /> },
    { path: "/login", element: <Login /> },
    { path: "/contact", element: <Contact /> }
]

export default routes;
