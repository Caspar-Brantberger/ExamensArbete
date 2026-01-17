import { useEffect,useState } from "react";
import './index.css';

interface Nurse {
    id: string;
    first_name: string;
    last_name: string;
    location?: string | null;
    skills?: string | null;

}

interface Recommendation {
    shift_id: string;
    score: number;
}

interface Shift {
    id:string;
    unique_shift_id: string;
    hospital_name: string;
    description: string;
    shift_start_date: string;
    shift_end_date: string;
    location: string;
    shift_type: string;
}


export default function RecommendationDemo() {
    const [nurses,setNurses] = useState<Nurse[]>([])
    const [selectedNurseId,setSelectedNurseId] = useState<string>("")
    const [recommendations,setRecommendations] = useState<Recommendation[]>([])
    const [shifts,setShifts] = useState<Shift[]>([])
    const [loading,setLoading] = useState<boolean>(false)

    useEffect(() => {
        fetch("http://127.0.0.1:8000/internal/nurses")
        .then((res) => res.json())
        .then((data) => setNurses(data))
        .catch((err) => console.error("Error fetching nurses",err))
    }, [])

    useEffect(() => {
    fetch("http://127.0.0.1:8000/internal/shifts")
        .then(res => res.json())
        .then(data => setShifts(data))
        .catch(err => console.error("Error fetching shifts", err));
        }, []);

    const generate = async () => {
        if(!selectedNurseId) return;
        setLoading(true);

        try {

            const TOKEN = "<your_api_token_here>"; // Replace with your actual token
            
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

        <div className="bg-gray-100 p-4 rounded space-y-2">
  <p className="font-semibold">Demo: ML-powered nurse shift recommendations</p>
  <p>This demo shows top shift recommendations for nurses based on a LightGBM model.</p>
  <p><span className="font-bold">Note:</span> All data is synthetic/fake, generated for demo purposes.</p>
  <p>Replace <code>&lt;your_api_token_here&gt;</code> with a valid JWT token to fetch recommendations. You can generate it with the backend script:
  python -m app.scripts.generate_mock_token </p>
  <p>The program is functional but real predictions require real data.</p>
</div>

      {/* Dropdown */}
        <select
        className="border p-2 w-full rounded"
        value={selectedNurseId}
        onChange={(e) => setSelectedNurseId(e.target.value)}
        >
        <option value="">Choose nurse </option>
        {nurses.map((n) => (
            <option key={n.id} value={n.id}>
            {n.first_name} {n.last_name}
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
            {recommendations.map((r) => {
        const shift = shifts.find(s => s.id === r.shift_id);
        if (!shift) return null; 
            return (
        <div key={r.shift_id} className="border p-3 rounded flex justify-between">
            <span>
                [{shift.unique_shift_id}] {shift.hospital_name} - {shift.shift_type} (
                {new Date(shift.shift_start_date).toLocaleString()} - {new Date(shift.shift_end_date).toLocaleString()}
                )
            </span>
            <div className="score-container">
        <div className="score-bar" style={{ width: `${r.score * 100}%` }}></div>
        <span className="score-text">{(r.score * 100).toFixed(1)}%</span>
        </div>
        </div>
                );
            })}
        </div>
        <p className="text-gray-500 text-sm mt-4">
    Disclaimer: This project was done with permission from the company I made this for due to them going bankrupt.
    </p>
    </div>
    );
}