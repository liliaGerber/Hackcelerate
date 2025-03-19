import apiClient from "./apiClient";

export const fetchHello = async (data) => {
    try{
        const response = await apiClient.get("/", data);
        return response.data;
    } catch (error) {
        console.error("API error:", error);
        throw error;
    }
}
