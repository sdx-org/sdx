import React, { useMemo } from 'react';
import { Alert } from 'react-bootstrap';

/** stringifySafe */
function stringifySafe(value) {
    try {
        return JSON.stringify(value, null, 2);
    } catch {
        return 'Unable to display wearable data';
    }
}

export default function WearableDataTab({ data }) {
    const wearableData = data?.data;
    const skipped = data?.skipped;
    /** Avoid unnecessary stringify when skipped or no data */
    const prettyWearable = useMemo(() => {
        if (skipped || !wearableData) return null;
        return stringifySafe(wearableData);
    }, [skipped, wearableData]);

    if (!data) return <p className="text-muted">No data available</p>;

    if (skipped) {
        return (
            <Alert variant="info">
                <strong>Skipped:</strong> User skipped uploading wearable data
            </Alert>
        );
    }

    const isEmpty =
        wearableData == null ||
        (Array.isArray(wearableData) && wearableData.length === 0) ||
        (!Array.isArray(wearableData) &&
            typeof wearableData === 'object' &&
            Object.keys(wearableData).length === 0);

    if (isEmpty) {
        return <p className="text-muted">No wearable data available</p>;
    }

    return (
        <div>
            <p>
                <strong>Wearable Data:</strong>
            </p>
            <div className="bg-light p-3 rounded">
                <pre style={{ fontSize: '0.85rem' }}>{prettyWearable}</pre>
            </div>
        </div>
    );
}
