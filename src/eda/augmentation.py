"""Agent-authored synthetic paraphrases; deterministic conservative word typos."""
import re

from .preprocessing import terms

PARAPHRASES = {
 'TRAIN_Q001': 'In Netcool/Impact (all versions), what should Exit() do inside User Defined Functions? Calling it from a User Defined Function in a Policy does not stop the Policy process.',
 'TRAIN_Q003': 'How can we enable SSL mutual authentication in IBM HTTP Server? Our application uses IHS v7 and needs this for a webservice. Where are the IHS setup instructions?',
 'TRAIN_Q005': 'Where is the load.rules FAQ maths function example rules file? It appears to have disappeared.',
 'TRAIN_Q006': 'Does the Apache CXF vulnerability CVE-2017-3156 affect ITNM? Which ITNM versions are vulnerable, and where are the details?',
 'TRAIN_Q008': 'Can the Requisite Pro (ReqPro) feature/plugin be used with a 64-bit RAD/RSA(4WS) 851+ installation, specifically a 64-bit RAD/RSA(4WS) 851 install?',
 'TRAIN_Q009': 'In Content Engine 5.2.0.3, where is the More tab under Document class - Property definition tab - property template configuration? I need to change Is Required.',
 'TRAIN_Q011': 'I bought a 12 month SPSS 25 license on Jan 27 and used it on my MacBook Pro. On a new MacBookPro, the downloaded program says I have no license, although I have the activation code. How can I use it?',
 'TRAIN_Q012': 'Where are the recommended ulimit settings for WebSphere on Linux documented?',
 'TRAIN_Q013': 'Is it possible to install a TIP 2.2 fix pack directly over TIP 2.1?',
 'TRAIN_Q016': 'How should I resolve "Too many open files" messages in the DASH systemOut log?',
 'TRAIN_Q017': 'We use Portal Server v6.1.0.6 / WPS 6.1.0.6 (Portal 6.1.x). Is Oracle 12c supported? If not, what is the newest supported Oracle 11g version?',
 'TRAIN_Q018': 'Why does deleting a virtual portal produce a transaction timeout when Managed pages is enabled?',
 'TRAIN_Q026': 'Where can I download the ITM VMware VI Agent Reports package for v 7.2? I only find ver 6x and v7.2 searches lead to https://www.ibm.com/developerworks/community/wikis/home?lang=en#!/wiki/W22ac5f3ebabf_4ff4_bd55_1a897c9a443e/page/Report%20Catalog, which lacks v 7.2.',
 'TRAIN_Q028': 'Where can I find an explanation of the Security Bulletin about Apache Commons FileUpload Vulnerabilities in IBM WebSphere MQ File Transfer Edition component (CVE-2016-1000031)?',
 'TRAIN_Q030': 'Can DataPower use SHA-2?',
 'TRAIN_Q031': 'How can I fix the DataPower error "non idempotent request method - RFC 2616 sec 10.3." when a POST request is redirected through DataPower?',
 'TRAIN_Q032': 'Why does a WCM User get com.ibm.icm.jcr.access.AccessDeniedException in SystemOut.log while trying to access authorized content?',
 'TRAIN_Q033': 'Why do scheduled reports stop working after a password change?',
 'TRAIN_Q034': 'Is ITNM 4.2 Fix Pack 3 (FP3) available? Where is the download link and what is its build number?',
 'TRAIN_Q035': 'How can I measure the efficiency of DataPower remote log targets (remote endpoint)?'
}


def protected(text):
    # Conservative measurable entity subset; prose semantics require separate review.
    return set(re.findall(r'\b(?:CVE-\d+-\d+|\w+(?:[./_:-]\w+)+|[A-Z]{2,}[A-Za-z0-9-]*)\b', text))


def make_augmentations(qa):
    rows=[]
    by_id = {q['query_id']:q for q in qa}
    for qid, paraphrase in PARAPHRASES.items():
        q = by_id[qid]
        original = q['question']
        # Typo map only ordinary lowercase prose; product names, flags, numbers untouched.
        typo = re.sub(r'\b(settings|running|request|support|available|reports|function|version|password|installation|configuration|expected|purchased|requires|access|changing|tracking)\b',
                      lambda m: m[0][:1]+m[0][2]+m[0][1]+m[0][3:], original, count=1)
        if typo == original:
            typo = re.sub(r'\b(the|with|after|does|this|where)\b', lambda m:m[0][1]+m[0][0]+m[0][2:], original, count=1)
        for method,text in [('paraphrase',paraphrase),('typo',typo)]:
            missing = sorted(protected(original)-protected(text))
            rows.append(dict(original_id=qid, source_split=q['split'], original_id_prefix=q['original_id_prefix'],
                             method=method, seed=42, original=original, transformed=text,
                             provenance='synthetic_agent' if method=='paraphrase' else 'deterministic_adjacent_letter_swap',
                             parameters='manual paraphrase' if method=='paraphrase' else 'one whitelisted lowercase prose word',
                             entity_subset_preserved=not missing, missing_entities=missing,
                             semantic_review='agent draft; human unreviewed',
                             length_delta=len(text)-len(original)))
    return rows
