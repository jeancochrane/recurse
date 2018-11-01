import os
import string


def make_vocab():
    text = ''
    for fpath in os.listdir('data'):
        with open(os.path.join('data', fpath), encoding='utf-8') as f:
            text += f.read()
            text += '\n\n'

    # Remove unwanted punctuation.
    for char in string.punctuation + '—“”…':
        text = text.replace(char, '')

    text = text.replace('\n\n', ' ')
    text = text.split()

    # Preserve capitalization of proper names.
    proper_names = [
        'Adolphus', 'Alcibiades', 'Amblystomas', 'America', 'American',
        'Artaud', 'Athens', 'Augustine', 'Amici’s', 'Bachelard', 'Bachelard’s',
        'Bacon', 'Batencour', 'Baudelaire', 'Baudelairean', 'Baudelaire’s',
        'Bauzee', 'Bauzee’s', 'Bentham', 'Benthamite', 'Bentham’s', 'Berlinische',
        'Borges', 'Borges’s', 'Brazil', 'Burckhardt', 'Canguilhem', 'Cartesian',
        'Catherine', 'Catholic','Charenton', 'Chassaigne', 'China', 'Chinese',
        'Christ', 'Christian', 'Christianity', 'Church', 'Condillac’s', 'CounterReformation',
        'Cuvier’s', 'Darwin’s', 'Delamare', 'Deleule', 'Demia', 'Descartes',
        'Djerba', 'Emperor', 'Empire', 'England', 'English', 'Enlightenment',
        'Europe', 'European', 'Eusthenes', 'Eusthenes’', 'Fortbonnais', 'Fourier',
        'Fourth', 'France', 'Frankfurt', 'Frederick', 'French', 'Freud', 'Freud’s',
        'FunckBrentano', 'Galileo', 'Galileo’s', 'German', 'God', 'God’', 'Goya',
        'Great', 'GrecoRoman', 'Greece', 'Greek', 'Greeks', 'Guerry', 'Gustavus',
        'Guys', 'Guéroult', 'HETEROTOPIAS', 'Habermas', 'Haskala', 'He', 'Hegel',
        'Helvetius', 'Heterotopias', 'Hippocrates', 'Horkheimer', 'Howard', 'Hôpitaux', 'I', 'II',
        'Inquisition', 'I’d', 'I’m', 'Jesuit', 'Jesuits', 'Jewish', 'Juden', 'Julius', 'July',
        'Kant', 'Kantian', 'Kant’s', 'Keynes', 'King’s', 'Ledoux', 'Leon', 'Lessing',
        'Linnaeus', 'Logos', 'Loisel', 'Mallarmé', 'Marx', 'Marxism', 'Max',
        'Mendelssohn', 'Mendelssohn’s', 'Moses', 'Moslems', 'Napoleonic', 'Nietzsche', 'Nietzschean',
        'November', 'Oedipus', 'Orient', 'Paraguay', 'Paris',  'Persians', 'Phadon',
        'Pinel’s', 'Piranese’s', 'Places', 'Plato’s', 'Pliny', 'Polynesian',
        'Protestant', 'Puritan', 'Radzinovitz', 'Reformation', 'Renaissance',
        'Roussel', 'SaintBeuve', 'SaintLazare', 'Scandinavian', 'September', 'Serres',
        'Socrates', 'Stalinism', 'Stalinists', 'Talleyrand’s', 'Tournefort',
        'Vaux’s', 'Veron', 'Versailles', 'Vico', 'Weber', 'West', 'Western',
        'William', 'William', 'Zuni'
    ]

    for i, word in enumerate(text):
        new_word = word

        # Enforce lowercase for all words that are not marked as proper nouns.
        if word not in proper_names:
            new_word = word.lower()

        # Strip excess unnecessary punctuation.
        if new_word.startswith('‘') or new_word.endswith('’'):
            new_word = new_word.replace('‘', '').replace('’', '')

        text[i] = new_word

    chars = sorted(list(set(text)))

    return chars, text


if __name__ == '__main__':
    import sys
    import json

    target_file = sys.argv[1]

    chars, _ = make_vocab()

    char_dict = {'vocab': list(chars)}

    with open(target_file, 'w') as outfile:
        json.dump(char_dict, outfile)
