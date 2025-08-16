import React, { useState, useEffect } from 'react';
import './App.css';

// --- Configuration ---
const API_BASE_URL = 'http://localhost:8000'; // Your FastAPI backend URL

function App() {
  const [calls, setCalls] = useState([]);
  const [selectedCall, setSelectedCall] = useState(null);
  const [callDetails, setCallDetails] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch the list of all analyzed calls on component mount
  useEffect(() => {
    const fetchCalls = async () => {
      try {
        setLoading(true);
        const response = await fetch(`${API_BASE_URL}/calls`);
        if (!response.ok) {
          throw new Error('Failed to fetch calls');
        }
        const data = await response.json();
        setCalls(data);
        setError(null);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchCalls();
  }, []);

  // Fetch details when a call is selected
  useEffect(() => {
    if (!selectedCall) return;

    const fetchCallDetails = async () => {
      try {
        setLoading(true);
        const response = await fetch(`${API_BASE_URL}/calls/${selectedCall}`);
        if (!response.ok) {
          throw new Error('Failed to fetch call details');
        }
        const data = await response.json();
        setCallDetails(data);
        setError(null);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchCallDetails();
  }, [selectedCall]);

  return (
    <div className="flex h-screen bg-gray-100 font-sans">
      {/* Sidebar */}
      <div className="w-1/4 bg-white border-r border-gray-200">
        <div className="p-4 border-b">
          <h1 className="text-xl font-bold text-gray-800">Analyzed Calls</h1>
        </div>
        <div className="overflow-y-auto">
          {loading && calls.length === 0 && <p className="p-4">Loading calls...</p>}
          {error && <p className="p-4 text-red-500">{error}</p>}
          <ul>
            {calls.map((call) => (
              <li
                key={call.fileName}
                className={`p-4 cursor-pointer hover:bg-gray-50 ${selectedCall === call.fileName ? 'bg-blue-100 text-blue-700' : ''}`}
                onClick={() => setSelectedCall(call.fileName)}
              >
                {call.fileName}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Main Content */}
      <div className="w-3/4 p-6 overflow-y-auto">
        {loading && <p>Loading details...</p>}
        {!selectedCall && !loading && (
          <div className="flex items-center justify-center h-full">
            <p className="text-gray-500">Select a call to view details</p>
          </div>
        )}
        
        {callDetails && (
          <div>
            <h2 className="text-2xl font-bold mb-4 text-gray-800">{callDetails.fileName}</h2>
            
            {/* Lead Score & Sentiment */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              <div className="bg-white p-4 rounded-lg shadow">
                <h3 className="font-bold text-gray-600">Lead Score</h3>
                <p className={`text-3xl font-bold ${callDetails.leadScore.score === 'Hot' ? 'text-green-500' : 'text-yellow-500'}`}>
                  {callDetails.leadScore.score}
                </p>
                <p className="text-sm text-gray-500">Confidence: {(callDetails.leadScore.confidence * 100).toFixed(2)}%</p>
              </div>
              <div className="bg-white p-4 rounded-lg shadow">
                 <h3 className="font-bold text-gray-600">Overall Sentiment</h3>
                 <p className={`text-3xl font-bold ${callDetails.sentiment > 0 ? 'text-green-500' : 'text-red-500'}`}>
                    {callDetails.sentiment > 0.1 ? 'Positive' : callDetails.sentiment < -0.1 ? 'Negative' : 'Neutral'}
                 </p>
                 <p className="text-sm text-gray-500">Polarity: {callDetails.sentiment.toFixed(2)}</p>
              </div>
            </div>

            {/* "Wow" Moments */}
            <div className="bg-white p-4 rounded-lg shadow mb-6">
              <h3 className="font-bold text-gray-600 mb-2">"Wow" Moments</h3>
              {callDetails.wowMoments.length > 0 ? (
                <ul>
                  {callDetails.wowMoments.map((moment, index) => (
                    <li key={index} className="mb-2 p-2 border-l-4 border-blue-500 bg-blue-50 rounded">
                      <p className="font-semibold text-blue-800">Keyword: "{moment.keyword}"</p>
                      <p className="text-sm text-gray-700 italic">"...{moment.context}..."</p>
                    </li>
                  ))}
                </ul>
              ) : <p className="text-gray-500">No "wow" moments detected.</p>}
            </div>

            {/* Keywords and Topics */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                <div className="bg-white p-4 rounded-lg shadow">
                    <h3 className="font-bold text-gray-600 mb-2">Keywords</h3>
                    <div className="flex flex-wrap gap-2">
                        {callDetails.keywords.map(kw => <span key={kw} className="bg-gray-200 text-gray-800 text-xs font-semibold mr-2 px-2.5 py-0.5 rounded-full">{kw}</span>)}
                    </div>
                </div>
                <div className="bg-white p-4 rounded-lg shadow">
                    <h3 className="font-bold text-gray-600 mb-2">Topics</h3>
                     <div className="flex flex-wrap gap-2">
                        {callDetails.topics.map(topic => <span key={topic} className="bg-teal-100 text-teal-800 text-xs font-semibold mr-2 px-2.5 py-0.5 rounded-full">{topic}</span>)}
                    </div>
                </div>
            </div>

            {/* Transcript */}
            <div className="bg-white p-4 rounded-lg shadow">
              <h3 className="font-bold text-gray-600 mb-2">Transcript</h3>
              <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">{callDetails.transcript}</p>
            </div>

          </div>
        )}
      </div>
    </div>
  );
}

export default App;
