import unittest
import process_data
import os

class TestProcessData(unittest.TestCase):
    def test_load_files(self):
        """load_file loads the content and labels it correctly."""
        dirname = os.path.dirname(os.path.abspath(__file__))
        labeled_paths = [
            (os.path.join(dirname, "spam"), True),
            (os.path.join(dirname, "ham"), False),
        ]

        labeled_files = list(process_data.load_files(labeled_paths))
        self.assertEqual("this is spam", labeled_files[0][0])
        self.assertTrue(labeled_files[0][1])
        self.assertEqual("this is ham", labeled_files[1][0])
        self.assertFalse(labeled_files[1][1])

    def test_process_file_all_false(self):
        """If all input parameters are false, process_file doesn't change the content string"""

        content = """Die Bremer haben die SPD  mehr als 31 Prozent der Stimmen
        als stärkste Kraft gewählt.
        Die CDU kommt auf mehr als 17 Prozent,
        die Grünen auf fast 21 Prozent."""

        expected = content
        actual = process_data.process_file(content)

        self.assertEqual(expected, actual)

    def test_process_file_stirp_headers(self):
        """process_file will strip the headers correctly"""

        filepath = os.path.join(
            os.path.dirname(__file__),
            "mail.txt")
        with open(filepath, encoding="iso-8859-1") as f:
            content = f.read()

            expected = """Die Bremer haben die SPD mit mehr als 31 Prozent der Stimmen als stärkste Kraft gewählt. Die CDU 
kommt auf mehr als 17 Prozent, die Grünen auf fast 21 Prozent.
Während SPD und CDU im Bund nur knapp auseinander liegen, sieht die Lage im Land Bremen anders 
aus. Wie die Landeswahlleitung nach der Auszählung in der Nacht zu Montag im Internet bekannt gab, 
hat die SPD 31,47 Prozent der Zweitstimmen erhalten und liegt damit klar vor der CDU (17,25 Prozent). 
Die Grünen kommen auf mehr als 20,83 Prozent der Stimmen, die FDP auf 9,3 Prozent. Anders als im 
Bund liegen die Linken problemlos über der Fünf-Prozent-Hürde.
In den beiden Wahlkreisen im Land Bremen haben die Sozialdemokratin Sarah Ryglewski und ihr 
Bremerhavener Parteigenosse Uwe Schmidt ihre Direktmandate bei der Bundestagswahl verteidigt. Im 
Land Bremen haben von 459.736 Wahlberechtigten insgesamt 330.140 Personen gewählt; das entspricht 
einer Wahlbeteiligung von 71,81 Prozent.
"""
            actual = process_data.process_file(content, strip_header=True)
            self.assertEqual(expected, actual)

    def test_process_lower(self):
        """process_file will convert the content to lowercase."""

        content = """Die Bremer haben die SPD  mehr als 31 Prozent der Stimmen
        als stärkste Kraft gewählt.
        Die CDU kommt auf mehr als 17 Prozent,
        die Grünen auf fast 21 Prozent."""

        expected = """die bremer haben die spd  mehr als 31 prozent der stimmen
        als stärkste kraft gewählt.
        die cdu kommt auf mehr als 17 prozent,
        die grünen auf fast 21 prozent."""
        actual = process_data.process_file(content, lowercase=True)

        self.assertEqual(expected, actual)

    def test_process_remove_punctuation(self):
        """process_file will remove the punctuation correctly."""

        content = """Die Bremer haben die SPD  mehr als 31 Prozent der Stimmen
        als stärkste Kraft gewählt.
        Die CDU kommt auf mehr als 17 Prozent,
        die Grünen auf fast 21 Prozent."""

        expected = """Die Bremer haben die SPD  mehr als 31 Prozent der Stimmen
        als stärkste Kraft gewählt
        Die CDU kommt auf mehr als 17 Prozent
        die Grünen auf fast 21 Prozent"""
        actual = process_data.process_file(content, remove_punctuation=True)

        self.assertEqual(expected, actual)

    def test_process_file_replace_urls(self):
        """process_file will replace the URLs with a placeholder."""

        content = """https://www.butenunbinnen.de/nachrichten/politik/bundestagswahl-ergebnisse-bremen-100.html
        Die Bremer haben die SPD  mehr als 31 Prozent der Stimmen
        als stärkste Kraft gewählt
        Die CDU kommt auf mehr als 17 Prozent
        die Grünen auf fast 21 Prozent.
        S.a. https://www.butenunbinnen.de/nachrichten/gesellschaft/wahlkreis-ergebnisse-bremerhaven-bremen-100.html"""

        expected = """URL
        Die Bremer haben die SPD  mehr als 31 Prozent der Stimmen
        als stärkste Kraft gewählt
        Die CDU kommt auf mehr als 17 Prozent
        die Grünen auf fast 21 Prozent.
        S.a. URL"""
        actual = process_data.process_file(content, replace_urls=True)

        self.assertEqual(expected, actual)

    def test_process_file_replace_numbers(self):
        """process_file will replace the numbers with a placeholder."""

        content = """
        Die Bremer haben die SPD mit mehr als 31 Prozent der Stimmen als stärkste Kraft gewählt. Die CDU 
        kommt auf mehr als 17 Prozent, die Grünen auf fast 21 Prozent.
        Während SPD und CDU im Bund nur knapp auseinander liegen, sieht die Lage im Land Bremen anders 
        aus. Wie die Landeswahlleitung nach der Auszählung in der Nacht zu Montag im Internet bekannt gab, 
        hat die SPD 31,47 Prozent der Zweitstimmen erhalten und liegt damit klar vor der CDU (17,25 Prozent). 
        Die Grünen kommen auf mehr als 20,83 Prozent der Stimmen, die FDP auf 9,3 Prozent. Anders als im 
        Bund liegen die Linken problemlos über der Fünf-Prozent-Hürde.
        In den beiden Wahlkreisen im Land Bremen haben die Sozialdemokratin Sarah Ryglewski und ihr 
        Bremerhavener Parteigenosse Uwe Schmidt ihre Direktmandate bei der Bundestagswahl verteidigt. Im 
        Land Bremen haben von 459.736 Wahlberechtigten insgesamt 330.140 Personen gewählt; das entspricht 
        einer Wahlbeteiligung von 71,81 Prozent."""

        expected = """
        Die Bremer haben die SPD mit mehr als NUMBER Prozent der Stimmen als stärkste Kraft gewählt. Die CDU 
        kommt auf mehr als NUMBER Prozent, die Grünen auf fast NUMBER Prozent.
        Während SPD und CDU im Bund nur knapp auseinander liegen, sieht die Lage im Land Bremen anders 
        aus. Wie die Landeswahlleitung nach der Auszählung in der Nacht zu Montag im Internet bekannt gab, 
        hat die SPD NUMBER Prozent der Zweitstimmen erhalten und liegt damit klar vor der CDU (NUMBER Prozent). 
        Die Grünen kommen auf mehr als NUMBER Prozent der Stimmen, die FDP auf NUMBER Prozent. Anders als im 
        Bund liegen die Linken problemlos über der Fünf-Prozent-Hürde.
        In den beiden Wahlkreisen im Land Bremen haben die Sozialdemokratin Sarah Ryglewski und ihr 
        Bremerhavener Parteigenosse Uwe Schmidt ihre Direktmandate bei der Bundestagswahl verteidigt. Im 
        Land Bremen haben von NUMBER Wahlberechtigten insgesamt NUMBER Personen gewählt; das entspricht 
        einer Wahlbeteiligung von NUMBER Prozent."""
        actual = process_data.process_file(content, replace_numbers=True)

        self.assertEqual(expected, actual)

    def test_make_labeled_paths(self):
        """make_labeled_paths lists the content of a folder and labels it."""

        spam_dir = os.path.join(os.path.dirname(__file__), "..", "data", "spam")
        labeled_paths = process_data.make_labeled_paths(spam_dir, True)

        for filepath, label in labeled_paths:
            self.assertTrue(label)
            self.assertTrue(os.path.samefile(spam_dir, os.path.dirname(filepath)))

    def test_make_dataset(self):
        """make_dataset will create numpy arrays from the labeled paths."""

        spam_dir = os.path.join(os.path.dirname(__file__), "..", "data", "spam")
        labeled_paths = list(process_data.make_labeled_paths(spam_dir, True))
        labeled_files = list(process_data.load_files(labeled_paths))

        data = process_data.make_dataset(labeled_files)

        expected_len = len(labeled_paths)
        self.assertEqual(expected_len, len(data["spam"]))
        self.assertEqual(expected_len, len(data["mails"]))

        for label in data["spam"]:
            self.assertTrue(label)

        for filepath, mail in zip([fp for fp, _ in labeled_paths], data["mails"]):
            with (open(filepath, encoding="iso-8859-1")) as f:
                self.assertEqual(f.read(), mail)

if __name__ == '__main__':
    unittest.main()