from app.chat.engine import strip_markdown


def test_strips_headings_bold_and_bullets():
    md = (
        "Here's a comparison:\n\n### Owner\n- **Leave Policy**: People Operations\n"
        "* **Expense Policy**: Finance\n\n---\n\n## Notes:\nUse `HR portal`."
    )
    assert strip_markdown(md) == (
        "Here's a comparison:\n\nOwner:\n- Leave Policy: People Operations\n"
        "- Expense Policy: Finance\n\nNotes:\nUse HR portal."
    )


def test_leaves_plain_text_and_maths_alone():
    for text in ["The meal allowance is 75 USD per day.", "46 * 122 = 5612", "2*3*4 = 24", "- a\n- b"]:
        assert strip_markdown(text) == text


def test_links_keep_url():
    assert strip_markdown("See [the portal](https://hr.example).") == "See the portal (https://hr.example)."
