import React from 'react';
import { Alert } from 'react-bootstrap';

export default function WearableDataTab({ data }) {
    if (!data) return <p className="text-muted">No data available</p>;

    const { data: wearableData, skipped } = data;

    if (skipped) {
        return (
            <Alert variant="info">
                <strong>Skipped:</strong> User skipped uploading wearable data
            </Alert>
        );
    }

    if (!wearableData || wearableData.length === 0) {
        return <p className="text-muted">No wearable data available</p>;
    }

    return (
        <div>
            <p>
                <strong>Wearable Data:</strong>
            </p>
            <div className="bg-light p-3 rounded">
                <pre style={{ fontSize: '0.85rem' }}>
                    {JSON.stringify(wearableData, null, 2)}
                </pre>
            </div>
        </div>
    );
}
