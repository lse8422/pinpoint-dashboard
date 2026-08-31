// 기사 본문을 그 자리에서 긁어 임시 파일에 쓴다. 저장소에는 남기지 않는다.
//
// 공식 경로가 전부 막혔다.
//   네이버 검색 API — 신규 제휴를 받지 않는다고 공지되어 있다.
//   카카오 다음 검색 — 뉴스 검색이 없다. 웹문서·블로그·카페뿐이다.
//   빅카인즈 Open API — 로그인과 신청이 필요하고, 전재·복제·배포를 금한다.
//   공공데이터포털 — 주제별 정적 파일이라 화성시 실시간 수집에 못 쓴다.
//
// 그래서 구글 뉴스 링크를 브라우저로 열어 언론사 원문에 닿는 길만 남았다.
// 집에서 재보니 1,181건 중 1,124건(95%)에서 본문을 얻었다.
//
// 본문은 절대 저장소에 커밋하지 않는다. 남의 기사이기 때문이다.
// 판정에만 쓰고 버린다. 화면에 남는 것은 판정 결과와 우리가 쓴 요약뿐이다.
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

const BASE = __dirname;
const OUT = path.join(BASE, '_bodies_tmp.json');
const LIMIT = Number(process.env.EXP_LIMIT || 300);
const CONC = Number(process.env.EXP_CONC || 4);
// 러너가 느리면 페이지가 다 뜨기 전에 긁게 된다. 기다리는 시간을 밖에서 조절한다.
const WAIT = Number(process.env.EXP_WAIT || 2200);
const TIMEOUT = Number(process.env.EXP_TIMEOUT || 22000);

// 깃허브 러너에는 크롬이 깔려 있다. 없으면 환경변수로 받는다.
function chromePath() {
  if (process.env.CHROME_PATH && fs.existsSync(process.env.CHROME_PATH)) return process.env.CHROME_PATH;
  const cands = [
    '/usr/bin/google-chrome', '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium-browser', '/usr/bin/chromium',
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
  ];
  for (const c of cands) if (fs.existsSync(c)) return c;
  throw new Error('크롬을 찾지 못했습니다. CHROME_PATH 를 지정하십시오.');
}

const clean = t => (t || '').replace(/\s+/g, ' ').trim();

(async () => {
  const live = JSON.parse(fs.readFileSync(path.join(BASE, 'news_live.json'), 'utf8'));
  const items = live.items.filter(x => x.url).slice(-LIMIT);
  console.log('대상 %d건 · 동시 %d개 · 대기 %dms · 시간초과 %dms',
    items.length, CONC, WAIT, TIMEOUT);

  const exe = chromePath();
  console.log('크롬: ' + exe);
  const b = await puppeteer.launch({
    executablePath: exe,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu'],
  });

  const out = {};
  let n = 0, ok = 0;
  async function work(list) {
    const p = await b.newPage();
    await p.setViewport({ width: 1280, height: 900 });
    for (const it of list) {
      let body = '', host = '';
      try {
        await p.goto(it.url, { waitUntil: 'domcontentloaded', timeout: TIMEOUT });
        await new Promise(r => setTimeout(r, WAIT));
        host = new URL(p.url()).hostname;
        body = clean(await p.evaluate(() => {
          const sel = ['article', '#articleBody', '.article-body', '#news_body_area',
            '#articleBodyContents', '.news_body', '#article-view-content-div',
            '.article_body', '#dic_area', '.view_con', '#content'];
          for (const s of sel) {
            const e = document.querySelector(s);
            if (e && e.innerText.trim().length > 200) return e.innerText;
          }
          return document.body.innerText;
        // 판정에 넣는 것은 220자다. 300자만 담고 그 이상은 가져오지 않는다.
        })).slice(0, 300);
      } catch (e) { host = 'ERR'; }
      out[it.title] = { host, body };
      if (body.length >= 200) ok++;
      if (++n % 25 === 0) console.log('  %d / %d건 (본문 %d건)', n, items.length, ok);
    }
    await p.close();
  }

  const chunks = Array.from({ length: CONC }, () => []);
  items.forEach((it, i) => chunks[i % CONC].push(it));
  await Promise.all(chunks.map(work));
  await b.close();

  fs.writeFileSync(OUT, JSON.stringify(out));
  const pct = (ok / items.length * 100).toFixed(0);
  console.log('\n본문 %d / %d건 (%s%%)', ok, items.length, pct);
  console.log('임시 저장: _bodies_tmp.json — 커밋하지 않습니다.');

  // 데이터센터 주소에서 언론사가 막는지가 이번 실행의 진짜 시험이다.
  // 집에서는 95%였다. 여기서 크게 낮으면 상시 적용을 하면 안 된다.
  if (ok / items.length < 0.5) {
    console.log('\n※ 확보율이 절반에 못 미칩니다. 집에서 잰 95%와 크게 다릅니다.');
    console.log('   깃허브 주소가 막히는 것으로 보이며, 이 방식은 매일 실행에 넣으면 안 됩니다.');
  }
})();
