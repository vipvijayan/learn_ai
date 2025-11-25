import re
from bs4 import BeautifulSoup
import json
import logging

logger = logging.getLogger("student_report_extraction")

def extract_table_data(html):
    try:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            return []

        rows = table.find_all("tr")
        if not rows:
            return []

        # Build header subjects and their column spans (supports colspan)
        header_cells = rows[0].find_all("td")
        subjects = []
        for hc in header_cells:
            name = hc.get_text(strip=True)
            try:
                colspan = int(hc.get('colspan', 1))
            except Exception:
                colspan = 1
            subjects.append({"name": name, "colspan": colspan})

        # Flatten columns to make indexing simple
        total_cols = sum(s['colspan'] for s in subjects)

        # Find rows that contain expectations and completed values (by searching row text)
        expectation_cells = None
        completed_cells = None
        for r in rows:
            t = r.get_text(' ', strip=True)
            if 'District Expectation' in t:
                expectation_cells = r.find_all('td')
            if 'Completed this week' in t:
                completed_cells = r.find_all('td')

        # If not found, fall back to searching for specific keywords in all cells
        if expectation_cells is None:
            for r in rows:
                if any('District Expectation' in td.get_text() for td in r.find_all('td')):
                    expectation_cells = r.find_all('td')
                    break

        if completed_cells is None:
            for r in rows:
                if any('Completed this week' in td.get_text() for td in r.find_all('td')):
                    completed_cells = r.find_all('td')
                    break

        # Normalize cells lists to total_cols by repeating or trimming
        def normalize_cell_texts(cells):
            texts = [c.get_text(' ', strip=True) for c in cells]
            # If fewer than total_cols, try to split cells that include multiple labels
            if len(texts) < total_cols:
                new_texts = []
                for txt in texts:
                    # Split by two-line pattern if present
                    parts = [p for p in re.split(r'\n|\r|\s{2,}', txt) if p.strip()]
                    if parts:
                        new_texts.extend(parts)
                    else:
                        new_texts.append(txt)
                texts = new_texts
            # Trim or pad
            if len(texts) > total_cols:
                texts = texts[:total_cols]
            while len(texts) < total_cols:
                texts.append('')
            return texts

        exp_texts = normalize_cell_texts(expectation_cells) if expectation_cells else [''] * total_cols
        comp_texts = normalize_cell_texts(completed_cells) if completed_cells else [''] * total_cols

        # Map columns into subjects using colspan
        results = []
        idx = 0
        for subj in subjects:
            colspan = subj['colspan']
            slice_exp = exp_texts[idx: idx + colspan]
            slice_comp = comp_texts[idx: idx + colspan]

            # Attempt to find minutes vs lessons in expectation slice
            minutes_expected = None
            lessons_expected = None
            for s in slice_exp:
                s_lower = s.lower()
                if 'minute' in s_lower:
                    minutes_expected = re.sub(r'[^0-9]', '', s)
                elif 'lesson' in s_lower:
                    lessons_expected = re.sub(r'[^0-9]', '', s)

            # Attempt to find minutes vs lessons in completed slice
            minutes_completed = None
            lessons_completed = None
            for s in slice_comp:
                s_lower = s.lower()
                if 'minute' in s_lower:
                    minutes_completed = re.sub(r'[^0-9]', '', s)
                elif 'lesson' in s_lower:
                    lessons_completed = re.sub(r'[^0-9]', '', s)
                else:
                    # If pure number, decide by column position heuristics
                    maybe_num = re.sub(r'[^0-9]', '', s)
                    if maybe_num:
                        # First column often minutes, second lessons (heuristic)
                        if minutes_completed is None:
                            minutes_completed = maybe_num
                        else:
                            lessons_completed = maybe_num

            results.append({
                'subject_name': subj['name'],
                'minutes_expected': int(minutes_expected) if minutes_expected and minutes_expected.isdigit() else minutes_expected,
                'lessons_expected': int(lessons_expected) if lessons_expected and lessons_expected.isdigit() else lessons_expected,
                'minutes_completed': int(minutes_completed) if minutes_completed and minutes_completed.isdigit() else minutes_completed,
                'lessons_completed': int(lessons_completed) if lessons_completed and lessons_completed.isdigit() else lessons_completed
            })
            idx += colspan

        return results
    except Exception as e:
        logger.error(f"Table extraction error: {e}")
        return []

def regex_extract_student_report(body):
    subjects = []
    subject_patterns = re.findall(r'(Math|Reading|Science|English|Social Studies)\s+Engagement', body, re.IGNORECASE)
    for subj_name in set(subject_patterns):
        expected_match = re.search(r'District Expectation:\s*(\d+)\s*Lesson', body, re.IGNORECASE)
        expected = int(expected_match.group(1)) if expected_match else 0
        completed_match = re.search(r'Completed this week:\s*(\d+)', body, re.IGNORECASE)
        completed = int(completed_match.group(1)) if completed_match else 0
        subjects.append({
            "name": subj_name.capitalize(),
            "expected_lessons": expected,
            "completed_lessons": completed,
            "performance": "See email for details",
            "summary": f"{subj_name} report"
        })
    return {
        "subjects": subjects,
        "overall_summary": "Extracted using regex fallback"
    }

def parse_llm_response(response):
    json_start = response.find('{')
    json_end = response.rfind('}') + 1
    if json_start != -1 and json_end > json_start:
        json_str = response[json_start:json_end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from LLM response")
    return {}
