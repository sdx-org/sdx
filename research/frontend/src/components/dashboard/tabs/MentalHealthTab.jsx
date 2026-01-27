import React from 'react';

export default function MentalHealthTab({ data }) {
    if (!data?.mental_health) return <p className="text-muted">No data available</p>;

    return (
        <div>
            <p>
                <strong>Mental Health Notes:</strong>
            </p>
            <p className="bg-light p-3 rounded">{data.mental_health}</p>
        </div>
    );
}
