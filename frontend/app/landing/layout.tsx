export default function LandingLayout({
    children,
}: {
    children: React.ReactNode
}) {
    // This layout removes the dashboard navigation for marketing pages
    return <>{children}</>
}
