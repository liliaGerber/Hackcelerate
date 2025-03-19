import { useState } from "react";
import { AppBar, Toolbar, Typography, Button, IconButton, Drawer, List, ListItem, ListItemText } from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import { Link } from "react-router-dom";
import { useMonitorSize } from "../hooks/useMonitorSize";
import config from "../config";

const Navbar = () => {
    const { isMobile } = useMonitorSize();
    const [drawerOpen, setDrawerOpen] = useState(false);

    const links = [
        { name: "Home", path: "/" },
        { name: "Team", path: "/team" },
        { name: "Contact", path: "/contact" },
    ];

    return (
        <>
            {/* ✅ Full-Width Desktop Navbar (Fixed to Top) */}
            {!isMobile ? (
                <AppBar position="fixed" color="primary" elevation={3} sx={{ width: "100%" }}>
                    <Toolbar sx={{ display: "flex", justifyContent: "space-between", paddingX: 2 }}>
                        <Typography variant="h6">{config.companyName}</Typography>
                        <div>
                            {links.map((link) => (
                                <Button key={link.path} color="inherit" component={Link} to={link.path} sx={{ marginX: 1 }}>
                                    {link.name}
                                </Button>
                            ))}
                        </div>
                        <Button variant="contained" color="secondary" component={Link} to="/login">
                            Login
                        </Button>
                    </Toolbar>
                </AppBar>
            ) : (
                <AppBar position="fixed" color="primary" elevation={3} sx={{ width: "100%" }}>
                    <Toolbar sx={{ display: "flex", justifyContent: "space-between", paddingX: 2 }}>
                        <Typography variant="h6">{config.companyName}</Typography>
                        <IconButton onClick={() => setDrawerOpen(true)} edge="end" color="inherit">
                            <MenuIcon />
                        </IconButton>
                    </Toolbar>
                </AppBar>
            )}

            {/* ✅ Full-Screen Mobile Drawer */}
            <Drawer
                anchor="left"
                open={drawerOpen}
                onClose={() => setDrawerOpen(false)}
                sx={{
                    "& .MuiDrawer-paper": { width: "100vw", height: "100vh" }, // Full screen
                }}
            >
                <List>
                    {links.map((link) => (
                        <ListItem button key={link.path} component={Link} to={link.path} onClick={() => setDrawerOpen(false)}>
                            <ListItemText primary={link.name} />
                        </ListItem>
                    ))}
                    <ListItem button component={Link} to="/login" onClick={() => setDrawerOpen(false)}>
                        <ListItemText primary="Login" />
                    </ListItem>
                </List>
            </Drawer>

            {/* ✅ Push Content Down to Prevent Overlap */}
            <div style={{ height: "64px" }} />
        </>
    );
};

export default Navbar;
