import React, { useState } from "react";
import { getEcsMapping } from "../api";
import logo from "../assets/logo.png";

// ✅ Import ShadCN UI components
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";

const EcsMapping = () => {
    const [logField, setLogField] = useState(""); // Stores user JSON input
    const [ecsMappings, setEcsMappings] = useState([]); // Stores response
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null); // Stores error messages

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!logField.trim()) return;

        setLoading(true);
        setError(null);

        try {
            // ✅ Parse JSON input (if invalid, catch error)
            const parsedInput = JSON.parse(logField);

            // ✅ Extract keys as an array for API request
            const logFieldsArray = Object.keys(parsedInput);

            // ✅ Send extracted keys to the backend
            const result = await getEcsMapping(logFieldsArray);
            setEcsMappings(result); // ✅ Store response
        } catch (error) {
            console.error("Invalid JSON format:", error);
            setError("Invalid JSON format. Please enter valid JSON.");
            setEcsMappings([]);
        }

        setLoading(false);
    };

    return (
        <div className="flex flex-col items-center justify-center h-screen w-screen bg-gray-100 p-6">
            <Card className="w-full max-w-5xl h-[90vh] p-6 rounded-xl shadow-md bg-white flex flex-col justify-between">

                {/* ✅ Centered Logo Section */}
                <div className="flex justify-center items-center gap-6 pb-4">
                    {[...Array(5)].map((_, index) => (
                        <img key={index} src={logo} alt={`Logo ${index + 1}`} className="w-24 h-auto" />
                    ))}
                </div>

                <CardHeader className="text-left">
                    <CardTitle className="text-3xl font-bold text-blue-700">ECS Mapping</CardTitle>
                </CardHeader>

                <CardContent className="flex flex-col flex-grow">
                    {/* ✅ Error Alert */}
                    {error && (
                        <Alert variant="destructive" className="mb-4">
                            <AlertTitle>Error</AlertTitle>
                            <AlertDescription>{error}</AlertDescription>
                        </Alert>
                    )}

                    {/* ✅ Input Form */}
                    <form onSubmit={handleSubmit} className="flex flex-col gap-4 flex-grow">
                        <Textarea
                            value={logField}
                            onChange={(e) => setLogField(e.target.value)}
                            placeholder='Enter JSON (e.g. {"account_name": "steven_com", "department_id": 90})'
                            className="h-full min-h-[200px] max-h-[400px] font-mono resize-none"
                        />
                        <Button type="submit" className="h-12 text-lg text-white bg-blue-600 hover:bg-blue-700"
                                disabled={loading}>
                            {loading ? "Mapping..." : "Get ECS Mapping"}
                        </Button>
                    </form>


                    {/* ✅ Display Mappings */}
                    {ecsMappings.length > 0 && (
                        <div className="mt-6 p-4 bg-gray-200 rounded-md overflow-y-auto max-h-[300px]">
                            <h3 className="text-xl font-semibold mb-2">Mappings</h3>
                            <ul className="space-y-2">
                                {ecsMappings.map((mapping, index) => (
                                    <li key={index} className="flex justify-between border p-2 rounded-md bg-white">
                                        <span className="font-semibold">{mapping.logField}</span>
                                        <span>{mapping.ecsField}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    );
};

export default EcsMapping;
