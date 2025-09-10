import React, { useState, useEffect } from 'react';
import { getCallsList } from './services/api';
import CallList from './components/CallList';
import CallDetails from './components/CallDetails';

// A more descriptive name for the main application component
function VoiceLeadDashboard() {
  // State for the list of calls, the currently selected call's filename, and any errors
  const [callList, setCallList] = useState([]);
  const [selectedCallFileName, setSelectedCallFileName] = useState(null);
  const [error, setError] = useState(null);
  const [isLoadingList, setIsLoadingList] = useState(true);

  // Fetch the list of all analyzed calls when the component first mounts
  useEffect(() => {
    const fetchInitialCallList = async () => {
      try {
        setIsLoadingList(true);
        const data = await getCallsList();
        setCallList(data);
        setError(null);
      } catch (err) {
        setError('Failed to fetch the list of calls. Is the backend server running?');
        console.error(err);
      } finally {
        setIsLoadingList(false);
      }
    };
    fetchInitialCallList();
  }, []); // Empty dependency array ensures this runs only once on mount

  // Handler for when a user clicks on a call in the sidebar
  const handleSelectCall = (fileName) => {
    setSelectedCallFileName(fileName);
  };

  return (
    <div className="flex h-screen bg-gray-100 font-sans">
      {/* Sidebar for navigation */}
      <aside className="w-1/4 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-4 border-b">
          <h1 className="text-xl font-bold text-gray-800">Analyzed Calls</h1>
        </div>
        <CallList
          calls={callList}
          selectedCall={selectedCallFileName}
          onSelectCall={handleSelectCall}
          isLoading={isLoadingList}
          error={error}
        />
      </aside>

      {/* Main content area to display details */}
      <main className="w-3/4 p-6 overflow-y-auto">
        <CallDetails
          selectedCallFileName={selectedCallFileName}
        />
      </main>
    </div>
  );
}

export default VoiceLeadDashboard;
