import React from 'react';
import { Badge } from 'react-bootstrap';

// Helpers for stable keys and labels
const getDiagKey = (diag, idx) => {
    if (typeof diag === 'string') return `s:${diag}`;
    if (diag && (diag.id ?? diag.code ?? diag.name)) {
        const val = diag.id ?? diag.code ?? diag.name;
        return `o:${String(val)}`;
    }
    return `idx:${idx}`;
};

const getDiagLabel = (diag) => {
    if (typeof diag === 'string') return diag;
    return diag?.name ?? '';
};

export default function DiagnosisTab({ data }) {
    if (!data) return <p className="text-muted">No data available</p>;

    const { suggestions, summary, selected } = data;

    return (
        <div>
            {summary && (
                <div className="mb-3">
                    <strong>AI Summary:</strong>
                    <p className="bg-light p-3 rounded">{summary}</p>
                </div>
            )}

            {suggestions && suggestions.length > 0 && (
                <>
                    <p>
                        <strong>Suggestions:</strong>
                    </p>
                    <ul className="list-group mb-3">
                        {suggestions.map((diag, idx) => (
                            <li
                                key={getDiagKey(diag, idx)}
                                className="list-group-item"
                            >
                                {getDiagLabel(diag)}
                            </li>
                        ))}
                    </ul>
                </>
            )}

            {selected && selected.length > 0 && (
                <>
                    <p>
                        <strong>Selected by Physician:</strong>
                    </p>
                    <div className="d-flex flex-wrap gap-2">
                        {selected.map((diag, idx) => (
                            <Badge key={getDiagKey(diag, idx)} bg="primary">
                                {getDiagLabel(diag)}
                            </Badge>
                        ))}
                    </div>
                </>
            )}

            {!suggestions?.length && !selected?.length && !summary && (
                <p className="text-muted">No diagnosis data available</p>
            )}
        </div>
    );
}
