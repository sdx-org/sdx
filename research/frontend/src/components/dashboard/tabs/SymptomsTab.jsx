import React from 'react';

/** normalizeSymptoms

@param {unknown} symptoms
@returns {string}
*/
function normalizeSymptoms(symptoms) {
    if (Array.isArray(symptoms)) {
        return symptoms
            .map((s) => (typeof s === 'string' ? s.trim() : ''))
            .filter(Boolean)
            .join(', ');
    }
    if (typeof symptoms === 'string') {
        return symptoms.trim();
    }
    return '';
}

export default function SymptomsTab({ data }) {
    const symptomsText = normalizeSymptoms(data?.symptoms);
    if (!symptomsText) return <p className="text-muted">No data available</p>;

    return (
        <div>
            <p>
                <strong>Symptoms:</strong>
            </p>
            <p className="bg-light p-3 rounded">{symptomsText}</p>
        </div>
    );
}
