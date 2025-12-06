import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navigation from "@/components/Navigation";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
    title: "PCDS Enterprise - Network Detection & Response",
    description: "AI-powered Attack Signal Intelligence and Threat Hunting Platform",
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en">
            <body className={`${inter.className} bg-slate-950 text-slate-100`}>
                {children}
            </body>
        </html>
    );
}
