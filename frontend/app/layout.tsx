import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navigation from "@/components/Navigation";
import { ThemeProvider, ThemeToggle } from "@/components/ThemeProvider";
import { ToastProvider } from "@/components/ToastProvider";
import { ModalProvider } from "@/components/ModalProvider";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
    title: "PCDS Enterprise - Network Detection & Response",
    description: "AI-powered Attack Signal Intelligence and Threat Hunting Platform",
    icons: {
        icon: '/favicon.svg',
    },
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en">
            <body className={`${inter.className} bg-slate-950 text-slate-100`}>
                <ThemeProvider>
                    <ToastProvider>
                        <ModalProvider>
                            {children}
                        </ModalProvider>
                    </ToastProvider>
                </ThemeProvider>
            </body>
        </html>
    );
}
