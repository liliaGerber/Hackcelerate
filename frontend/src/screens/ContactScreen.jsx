import React, { useState } from "react";
import { TextField, Button, CircularProgress, Container, Stack, Typography, Box } from "@mui/material";
import emailjs from "emailjs-com";

const ContactForm = () => {
    const [form, setForm] = useState({
        name: "",
        email: "",
        phone: "",
        subject: "",
        message: "",
    });
    const [isValid, setIsValid] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [statusMessage, setStatusMessage] = useState("");

    const rules = {
        required: (value) => !!value || "This field is required.",
        email: (value) =>
            /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value) || "Invalid email address.",
    };

    const validateForm = () => {
        setIsValid(
            form.name &&
            form.email &&
            form.subject &&
            form.message &&
            rules.email(form.email) === true
        );
    };

    const handleChange = (e) => {
        setForm({ ...form, [e.target.name]: e.target.value });
        validateForm();
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        try {
            await emailjs.send(
                import.meta.env.VITE_EMAILJS_SERVICE_ID,
                import.meta.env.VITE_EMAILJS_TEMPLATE_ID,
                {
                    from_name: form.name,
                    from_email: form.email,
                    from_phone: form.phone,
                    subject: form.subject,
                    message: form.message,
                },
                import.meta.env.VITE_EMAILJS_PUBLIC_KEY
            );
            setStatusMessage("Message sent successfully!");
            setForm({ name: "", email: "", phone: "", subject: "", message: "" });
        } catch (error) {
            console.error("Email sending failed:", error);
            setStatusMessage("Error sending email. Please try again later.");
        }
        setIsLoading(false);
    };

    return (
        <Container maxWidth={false} className="min-h-screen flex justify-center items-center bg-primary-main px-4" style={{ width: "100vw" }}>
            <Box className="w-full max-w-2xl flex flex-col items-center">
                <Typography variant="h2" className="text-primary text-center font-bold mb-4">
                    Just say Hello!
                </Typography>
                <Box component="form" onSubmit={handleSubmit} className="space-y-8 w-full mt-5">
                    <Stack spacing={4} direction={{ xs: 'column', sm: 'row' }}>
                        <TextField
                            label="Name"
                            name="name"
                            value={form.name}
                            onChange={handleChange}
                            fullWidth
                            required
                            InputLabelProps={{ shrink: undefined }}
                            sx={{
                                backgroundColor: "rgba(255, 255, 255, 0.2)",
                                borderRadius: "12px",
                                input: { color: "white" },
                                '& .MuiInputLabel-root': { color: 'white' },
                            }}
                        />
                        <TextField
                            label="Subject"
                            name="subject"
                            value={form.subject}
                            onChange={handleChange}
                            fullWidth
                            required
                            InputLabelProps={{ shrink: undefined }}
                            sx={{
                                backgroundColor: "rgba(255, 255, 255, 0.2)",
                                borderRadius: "12px",
                                input: { color: "white" },
                                '& .MuiInputLabel-root': { color: 'white' },
                            }}
                        />
                    </Stack>
                    <Stack spacing={4} direction={{ xs: 'column', sm: 'row' }}>
                        <TextField
                            label="Mail"
                            name="email"
                            value={form.email}
                            onChange={handleChange}
                            fullWidth
                            required
                            InputLabelProps={{ shrink: undefined }}
                            sx={{
                                backgroundColor: "rgba(255, 255, 255, 0.2)",
                                borderRadius: "12px",
                                input: { color: "white" },
                                '& .MuiInputLabel-root': { color: 'white' },
                            }}
                        />
                        <TextField
                            label="Phone"
                            name="phone"
                            value={form.phone}
                            onChange={handleChange}
                            fullWidth
                            InputLabelProps={{ shrink: undefined }}
                            sx={{
                                backgroundColor: "rgba(255, 255, 255, 0.2)",
                                borderRadius: "12px",
                                input: { color: "white" },
                                '& .MuiInputLabel-root': { color: 'white' },
                            }}
                        />
                    </Stack>
                    <TextField
                        label="Message"
                        name="message"
                        value={form.message}
                        onChange={handleChange}
                        fullWidth
                        required
                        multiline
                        minRows={5}
                        InputLabelProps={{ shrink: undefined }}
                        sx={{
                            backgroundColor: "rgba(255, 255, 255, 0.2)",
                            borderRadius: "12px",
                            textarea: { color: "white" },
                            '& .MuiInputLabel-root': { color: 'white' },
                        }}
                    />
                    <Box className="mt-6">
                        <Button
                            type="submit"
                            fullWidth
                            sx={{ backgroundColor: "primary.main", color: "white", padding: "14px", fontWeight: "bold", borderRadius: "20px" }}
                            disabled={!isValid || isLoading}
                        >
                            {isLoading ? <CircularProgress size={24} sx={{ color: "white" }} /> : "SUBMIT"}
                        </Button>
                    </Box>
                    {statusMessage && <p className="text-green-500 mt-2 text-center">{statusMessage}</p>}
                </Box>
            </Box>
        </Container>
    );
};

export default ContactForm;
