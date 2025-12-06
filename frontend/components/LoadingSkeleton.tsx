'use client';

export function LoadingSkeleton({ className = "" }: { className?: string }) {
    return (
        <div className={`animate-pulse space-y-4 ${className}`}>
            <div className="h-4 bg-slate-700 rounded w-3/4"></div>
            <div className="h-4 bg-slate-700 rounded"></div>
            <div className="h-4 bg-slate-700 rounded w-5/6"></div>
        </div>
    );
}

export function CardSkeleton() {
    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 animate-pulse">
            <div className="h-6 bg-slate-700 rounded w-1/3 mb-4"></div>
            <div className="space-y-3">
                <div className="h-4 bg-slate-700 rounded"></div>
                <div className="h-4 bg-slate-700 rounded w-5/6"></div>
                <div className="h-4 bg-slate-700 rounded w-4/6"></div>
            </div>
        </div>
    );
}

export function TableSkeleton({ rows = 5 }: { rows?: number }) {
    return (
        <div className="space-y-2">
            {[...Array(rows)].map((_, i) => (
                <div key={i} className="flex gap-4 animate-pulse">
                    <div className="h-10 bg-slate-700 rounded flex-1"></div>
                    <div className="h-10 bg-slate-700 rounded w-24"></div>
                    <div className="h-10 bg-slate-700 rounded w-32"></div>
                </div>
            ))}
        </div>
    );
}

export function ChartSkeleton({ height = "300px" }: { height?: string }) {
    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
            <div className="h-6 bg-slate-700 rounded w-1/3 mb-4 animate-pulse"></div>
            <div
                className="bg-slate-800 rounded animate-pulse"
                style={{ height }}
            />
        </div>
    );
}

export function StatCardSkeleton() {
    return (
        <div className="bg-gradient-to-br from-slate-800 to-slate-900 p-6 rounded-xl border border-slate-700 animate-pulse">
            <div className="flex items-center justify-between mb-4">
                <div className="h-8 w-8 bg-slate-700 rounded-full"></div>
            </div>
            <div className="h-4 bg-slate-700 rounded w-1/2 mb-2"></div>
            <div className="h-8 bg-slate-700 rounded w-3/4"></div>
        </div>
    );
}
