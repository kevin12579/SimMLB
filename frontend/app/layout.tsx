import type { Metadata } from 'next'
import { Barlow_Condensed, JetBrains_Mono } from 'next/font/google'
import './globals.css'

const barlow = Barlow_Condensed({
  subsets: ['latin'],
  weight: ['400', '600', '700'],
  variable: '--font-barlow',
  display: 'swap',
})

const jetbrains = JetBrains_Mono({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-mono-jb',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'Diamond Lines · MLB AI 승부예측',
  description: 'MLB 정규시즌 당일 경기 홈팀 승리 확률 예측 — LightGBM + XGBoost 앙상블',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body className={`${barlow.variable} ${jetbrains.variable}`}>
        {children}
      </body>
    </html>
  )
}
