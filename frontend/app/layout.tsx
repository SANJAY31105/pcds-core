import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navigation from "@/components/Navigation";
import { ThemeProvider, ThemeToggle } from "@/components/ThemeProvider";
import { ToastProvider } from "@/components/ToastProvider";
import { ModalProvider } from "@/components/ModalProvider";
import { AuthProvider } from "@/lib/AuthContext";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
    title: "PCDS Enterprise - Network Detection & Response",
    description: "AI-powered Attack Signal Intelligence and Threat Hunting Platform",
    manifest: '/manifest.json',
    themeColor: '#0a0a0a',
    viewport: {
        width: 'device-width',
        initialScale: 1,
        maximumScale: 1,
        userScalable: false, // Feels native
    },
    appleWebApp: {
        capable: true,
        statusBarStyle: 'black-translucent',
        title: 'PCDS',
    },
    icons: {
        icon: '/favicon.svg',
        apple: '/favicon.svg', // Ideally apple-touch-icon.png
    },
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en">
            <head>
                {/* PostHog Analytics */}
                <script
                    dangerouslySetInnerHTML={{
                        __html: `
                            !function(t,e){var o,n,p,r;e.__SV||(window.posthog=e,e._i=[],e.init=function(i,s,a){function g(t,e){var o=e.split(".");2==o.length&&(t=t[o[0]],e=o[1]),t[e]=function(){t.push([e].concat(Array.prototype.slice.call(arguments,0)))}}(p=t.createElement("script")).type="text/javascript",p.async=!0,p.src=s.api_host.replace(".i.posthog.com","-assets.i.posthog.com")+"/static/array.js",(r=t.getElementsByTagName("script")[0]).parentNode.insertBefore(p,r);var u=e;for(void 0!==a?u=e[a]=[]:a="posthog",u.people=u.people||[],u.toString=function(t){var e="posthog";return"posthog"!==a&&(e+="."+a),t||(e+=" (stub)"),e},u.people.toString=function(){return u.toString(1)+".people (stub)"},o="init capture register register_once register_for_session unregister unregister_for_session getFeatureFlag getFeatureFlagPayload isFeatureEnabled reloadFeatureFlags updateEarlyAccessFeatureEnrollment getEarlyAccessFeatures on onFeatureFlags onSessionId getSurveys getActiveMatchingSurveys renderSurvey canRenderSurvey identify setPersonProperties group resetGroups setPersonPropertiesForFlags resetPersonPropertiesForFlags setGroupPropertiesForFlags resetGroupPropertiesForFlags reset get_distinct_id getGroups get_session_id get_session_replay_url alias set_config startSessionRecording stopSessionRecording sessionRecordingStarted captureException loadToolbar get_property getSessionProperty createPersonProfile opt_in_capturing opt_out_capturing has_opted_in_capturing has_opted_out_capturing clear_opt_in_out_capturing debug getPageViewId captureTrackedForms captureIncrements".split(" "),n=0;n<o.length;n++)g(u,o[n]);e._i.push([i,s,a])},e.__SV=1)}(document,window.posthog||[]);
                            posthog.init('phc_PnubEaatO4Kzxp2NCiBWc1QYwgdTJEIZ52GKyalulap',{api_host:'https://us.i.posthog.com', person_profiles: 'always'})
                        `,
                    }}
                />
            </head>
            <body className={`${inter.className} bg-slate-950 text-slate-100`}>
                <ThemeProvider>
                    <AuthProvider>
                        <ToastProvider>
                            <ModalProvider>
                                {children}
                            </ModalProvider>
                        </ToastProvider>
                    </AuthProvider>
                </ThemeProvider>
            </body>
        </html>
    );
}
