'use client';

// Skeleton component for loading states
export function Skeleton({ className = '' }: { className?: string }) {
    return (
        <div className={`animate-pulse bg-[#1a1a1a] rounded ${className}`} />
    );
}

// Card skeleton
export function CardSkeleton() {
    return (
        <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
            <Skeleton className="h-4 w-24 mb-3" />
            <Skeleton className="h-8 w-16 mb-2" />
            <Skeleton className="h-3 w-20" />
        </div>
    );
}

// Table row skeleton
export function TableRowSkeleton({ cols = 5 }: { cols?: number }) {
    return (
        <tr className="border-b border-[#2a2a2a]">
            {Array.from({ length: cols }).map((_, i) => (
                <td key={i} className="px-4 py-4">
                    <Skeleton className="h-4 w-full" />
                </td>
            ))}
        </tr>
    );
}

// List item skeleton
export function ListItemSkeleton() {
    return (
        <div className="flex items-center gap-4 p-4 bg-[#141414] rounded-xl border border-[#2a2a2a]">
            <Skeleton className="w-10 h-10 rounded-lg" />
            <div className="flex-1">
                <Skeleton className="h-4 w-3/4 mb-2" />
                <Skeleton className="h-3 w-1/2" />
            </div>
            <Skeleton className="w-16 h-6 rounded" />
        </div>
    );
}

// Dashboard skeleton
export function DashboardSkeleton() {
    return (
        <div className="space-y-6">
            <div>
                <Skeleton className="h-8 w-48 mb-2" />
                <Skeleton className="h-4 w-64" />
            </div>
            <div className="grid grid-cols-4 gap-4">
                {Array.from({ length: 4 }).map((_, i) => (
                    <CardSkeleton key={i} />
                ))}
            </div>
            <div className="grid grid-cols-3 gap-6">
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                    <Skeleton className="h-4 w-32 mb-4" />
                    <div className="space-y-3">
                        {Array.from({ length: 4 }).map((_, i) => (
                            <div key={i}>
                                <Skeleton className="h-3 w-full mb-1" />
                                <Skeleton className="h-2 w-full" />
                            </div>
                        ))}
                    </div>
                </div>
                <div className="col-span-2 bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                    <Skeleton className="h-4 w-40 mb-4" />
                    <div className="space-y-2">
                        {Array.from({ length: 5 }).map((_, i) => (
                            <ListItemSkeleton key={i} />
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}

// Page header skeleton
export function PageHeaderSkeleton() {
    return (
        <div className="flex items-center justify-between mb-6">
            <div>
                <Skeleton className="h-8 w-48 mb-2" />
                <Skeleton className="h-4 w-64" />
            </div>
            <Skeleton className="h-10 w-32 rounded-lg" />
        </div>
    );
}
