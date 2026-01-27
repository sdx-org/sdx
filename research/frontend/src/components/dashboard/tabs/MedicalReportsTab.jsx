import React from 'react';
import { Alert } from 'react-bootstrap';

export default function MedicalReportsTab({ data }) {
    if (!data) return <p className="text-muted">No data available</p>;

    const { files, skipped } = data;

    if (skipped) {
        return (
            <Alert variant="info">
                <strong>Skipped:</strong> User skipped uploading medical reports
            </Alert>
        );
    }

    if (!files || files.length === 0) {
        return <p className="text-muted">No files uploaded</p>;
    }

    return (
        <div>
            <p>
                <strong>Medical Reports:</strong>
            </p>
            <ul className="list-group">
                {files.map((file, idx) => (
                    <li key={idx} className="list-group-item d-flex justify-content-between align-items-center">
                        <span>📄 {file.name || `File ${idx + 1}`}</span>
                        <span className="badge bg-secondary rounded-pill">{file.type}</span>
                    </li>
                ))}
            </ul>
        </div>
    );
}
