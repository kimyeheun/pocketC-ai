from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Tuple, List

# NOTE: 분류 결과 표현
@dataclass(frozen=True)
class CategoryHit:
    sub_name: str
    source: str

@dataclass(frozen=True)
class SubCategory:
    sub_id: int
    major_id: int
    sub_name: str

# NOTE: 정규식 기반 1차 룰
REGEX_RULES: List[Tuple[re.Pattern, str]] = [
    # 식비 (Food & Drink)
    (re.compile(r"스타벅스|이디야|투썸|메가|빽"), "커피"),
    (re.compile(r"(맥도날드|버거킹|롯데리아|KFC|BKR|치킨|치퀸|김밥|돈까스|족발|냉면|순대국|칼국수|비빔밥|식당|한식|중식|일식|양식|분식|푸드|오리바베|쌈|LABAB|왕돈까스|밀향기)"), "외식"),
    (re.compile(r"(요기요|배달의민족|쿠팡이츠)"), "배달음식"),
    (re.compile(r"(마켓컬리|홈플러스|이마트|롯데마트|식자재|축산|슈퍼|마트|식재료|프레시)"), "식재료"),
    (re.compile(r"(GS25|씨유|CU|세븐일레븐|이마트24|지에스|홈마트|편의점)"), "간식"),

    # 교통/차량
    (re.compile(r"(티머니|T-money|지하철|도시철도|코레일|교통|버스)"), "대중교통"),
    (re.compile(r"(택시|카카오T|유페이개인택시|대리)"), "택시/대리"),
    (re.compile(r"(주유소|GS칼텍스|SK에너지|현대오일뱅크|S-?OIL|오일|에너지)"), "유류비"),

    # 생활/쇼핑
    (re.compile(r"(다이소|아트박스|스타안경|생활용품)"), "생활용품"),
    (re.compile(r"프린트|프린팅|교보|문고|문구|도서|책"), "도서/교재"),

    # 통신/인터넷
    (re.compile(r"SKT|KT|LGU\+|통신"), "휴대폰 요금"),

    # OTT/구독
    (re.compile(r"넷플릭스|GPT|디즈니|티빙|왓챠|유튜브|웨이브|구글플레이|OPEN|CLAUDE"), "OTT/구독서비스"),

    # 쇼핑
    (re.compile(r"쿠팡|G마켓|11번가|SSG"), "온라인 쇼핑몰"),
    (re.compile(r"무신사|백화점|지그재그|ZIGZAG|알리|더현대서울|타임스퀘어"), "의류/패션"),
    (re.compile(r"올리브영|미용|머리|헤어|뷰티"), "뷰티/미용"),

    # 건강/의료
    (re.compile(r"(병원|의원|치과|한의원|정형|외과|내과)"), "병원"),
    (re.compile(r"약국"), "약국"),

    # 주거/공과금
    (re.compile(r"(관리비|임대료|월세)"), "월세/관리비"),
    (re.compile(r"(전기요금|한국전력|한전)"), "전기세"),
    (re.compile(r"(수도요금|수도사업본부)"), "수도세"),
    (re.compile(r"(가스비|도시가스)"), "가스비"),

    # 문화/여가
    (re.compile(r"(CGV|롯데시네마|메가박스|영화|팝콘)"), "영화/공연"),
    (re.compile(r"(스팀|Steam|닌텐도|플스|PlayStation|Xbox)"), "게임/콘텐츠"),
    (re.compile(r"(여기어때|NOL|Trip|Booking|아시아나|대한항공|제주항공|진에어|티웨이|부킹닷컴|Airbnb|호텔스닷컴)"), "여행"),
    (re.compile(r"(코인노|노래방|뽑기|인형|게임)"), "취미/오락"),

    # 금융/저축
    (re.compile(r"(국세청|지방세|세무서|건보공단|국민연금|고용보험|산재보험)"), "세금/보험"),
    (re.compile(r"(장학|대출|원리금|이자 상환|이자납부)"), "대출/이자"),
    (re.compile(r"(청약|적금|예금|자동이체 저축)"), "저축"),
    (re.compile(r"(증권|주식|펀드|ETF|코인|암호화폐|가상자산|키움)"), "투자"),

    # 기타
    (re.compile(r"(페이)"), "기타"),
]

# 식비 세분화 힌트
CAFE_HINT = re.compile(r"(공차|설빙|아마스빈|매머드|할리스|투썸|텐퍼센트|메가엠지씨|컴포즈|바나프레소|카페|커피|COFFEE|CAFE)")
CONVENIENCE_HINT = re.compile(r"(아이스크림|편의점|CU|GS25|세븐일레븐|지에스|씨유|이마트24)")
