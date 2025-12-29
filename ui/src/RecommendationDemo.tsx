import { useEffect,useState } from "react";

interface Nurse {
    id: string;
    name: string;
}

interface Recommendation {
    shift_id: string;
    score: number;
}


export default function RecommendationDemo() {
    const [nurses,setNurses] = useState<Nurse[]>([])
    const [selectedNurseId,setSelectedNurseId] = useState<string>("")
    const [recommendations,setRecommendations] = useState<Recommendation[]>([])
    const [loading,setLoading] = useState<boolean>(false)

    useEffect(() => {
        fetch("http://127.0.0.1:8000/internal/nurses")
        .then((res) => res.json())
        .then((data) => setNurses(data))
        .catch((err) => console.error("Error fetching nurses",err))
    }, [])

    const generate = async () => {
        if(!selectedNurseId) return;
        setLoading(true);

        try {

            const TOKEN = "your_api_token_here"; // Replace with your actual token
            
            const res = await fetch("http://127.0.0.1:8000/v1/recommendations/shifts",{
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${TOKEN}`
                },
                body: JSON.stringify({
                    nurse_id: selectedNurseId,
                    max_results:5,
                }),
            });
            if(!res.ok) {
                throw new Error('Error fetching recommendations');
        }
        const data: { nurse_id: string; recommendations: Recommendation[] } = await res.json();
        setRecommendations(data.recommendations);
    } catch (err) {
        console.error("Error generating recommendations",err);
        setRecommendations([]);
    }finally {
        setLoading(false);
    }
};
    return (
            <div className="max-w-xl mx-auto p-6 space-y-4">
        <h1 className="text-xl font-bold">Shift recommendations</h1>

      {/* Dropdown */}
        <select
        className="border p-2 w-full rounded"
        value={selectedNurseId}
        onChange={(e) => setSelectedNurseId(e.target.value)}
        >
        <option value="">Choose nurse </option>
        {nurses.map((n) => (
            <option key={n.id} value={n.id}>
            {n.name}
            </option>
        ))}
        </select>

      {/* Generate-Button */}
        <button
        className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
        disabled={!selectedNurseId || loading}
        onClick={generate}
        >
        {loading ? "Generating..." : "Generate"}
        </button>

      {/* Results */}
        <div className="space-y-2 mt-4">
        {recommendations.length === 0 && !loading && <p>No recommendations yet</p>}
        {recommendations.map((r) => (
            <div key={r.shift_id} className="border p-3 rounded flex justify-between">
            <span>{r.shift_id}</span>
            <span className="font-bold text-green-600">{(r.score * 100).toFixed(1)}%</span>
            </div>
        ))}
        </div>
    </div>
    );
}