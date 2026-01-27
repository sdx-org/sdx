import React from 'react';
import { Badge } from 'react-bootstrap';

export default function ExamsTab({ data }) {
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
                        <strong>Suggested Exams:</strong>
                    </p>
                    <ul className="list-group mb-3">
                        {suggestions.map((exam, idx) => (
                            <li key={idx} className="list-group-item">
                                {typeof exam === 'string' ? exam : exam.name}
                            </li>
                        ))}
                    </ul>
                </>
            )}

            {selected && selected.length > 0 && (
                <>
                    <p>
                        <strong>Selected Exams:</strong>
                    </p>
                    <div className="d-flex flex-wrap gap-2">
                        {selected.map((exam, idx) => (
                            <Badge key={idx} bg="success">
                                {exam}
                            </Badge>
                        ))}
                    </div>
                </>
            )}

            {!suggestions?.length && !selected?.length && !summary && (
                <p className="text-muted">No exam data available</p>
            )}
        </div>
    );
}
