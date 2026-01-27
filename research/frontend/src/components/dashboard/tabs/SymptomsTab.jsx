import React from 'react';

export default function SymptomsTab({ data }) {
    if (!data?.symptoms) return <p className="text-muted">No data available</p>;

    return (
        <div>
            <p>
                <strong>Symptoms:</strong>
            </p>
            <p className="bg-light p-3 rounded">{data.symptoms}</p>
        </div>
    );
}
