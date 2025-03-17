import axios from "axios";

const API_BASE_URL = "/api"; // ✅ Ensure this matches Django backend

export const getEcsMapping = async (logField) => {
    try {
        // Ensure logField is always an array
        const logFieldsArray = Array.isArray(logField) ? logField : [logField];


        const response = await axios.post(`${API_BASE_URL}/mappings/ecs/`, {
            log_field: logFieldsArray,  // ✅ Ensure correct field name
        }, {
            headers: { "Content-Type": "application/json" } // ✅ Explicitly set JSON headers
        });
        // Convert backend response to an array of objects
        const formattedResponse = Object.entries(response.data).map(([key, value]) =>({
            logField: key,
            ecsField: value !== "none_ecs_field" ? value : "No Mapping Found"
        }));

        return formattedResponse;
    } catch (error) {
        console.error("Error fetching ECS mapping:", error.response?.data || error.message);
        return { log_field: logField, ecs_field: "Error fetching mapping" };
    }
};



