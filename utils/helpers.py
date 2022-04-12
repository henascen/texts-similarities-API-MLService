from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import spacy
import numpy as np
import unicodedata
import re
from pathlib import Path


SPACY_MODEL_XX_PATH = Path.cwd() / 'xx_sent_ud_sm-3.2.0'


candidates_info_test = [
    {
        'name': 'candidateOne',
        'resume': 
            """contactar www.linkedin.com/in/vmreyesal (linkedin). 
            aptitudes principales react native, diseño web javascript
            languages español (native or bilingual) ingles (limited working)
            víctor m. reyes react developer en focus el salvador san salvador
            extracto desarrollador con capacidad analítica e investigativa,
            con más de siete años de experiencia en áreas de tecnología y seis
            años construyendo soluciones y servicios innovadores a través del 
            uso e implementación de nuevas tendencias tecnológicas.
            experiencia ito el salvador react native developer 
            marzo de 2022 - present (2 meses) san salvador, el salvador
            dev.f sensei ‍ (docente) mayo de 2021 - present (1 año) méxico
            business development group, s.a. react developer 
            septiembre de 2021 - marzo de 2022 (7 meses) guatemala  yeah 
            studio desarrollador web junio de 2020 - diciembre de 2021 
            (1 año 7 meses) el salvador citylab react native developer 
            enero de 2019 - mayo de 2021 (2 años 5 meses) san salvador 
            travelennial front-end developer octubre de 2016 - enero de 2019 
            (2 años 4 meses) educación universidad francisco gavidia 
            lic. sistemas de computación administrativa,
            desarrollo de aplicaciones web  (2006 - 2017) actividad 
            04/07/2022, visto por walter marroquin
        """
    },
    {
        'name': 'candidateTwo',
        'resume': (
            'Francisco Briceno FRANCISCO BRICEÑO Java Developer Summary .'
            ' 8+ years of experience working with Java. . Various database'
            ' engine technologies such as MS SQL, Postgres and MySQL. . '
            'Integración continua con Jenkins. . Version control software such'
            ' as GIT. . Agile / Scrum methodology and tools such as JIRA and '
            'Azure Boards. Work Experience BAC Credomatic Analista Programador '
            'Recional Senior 2019 2020 Análisis, desarrollo e implementación '
            'de soluciones para la banca en línea, realizando tareas de '
            'programación de nuevas funcionalidades, mejora continua del '
            'código, documentación de las soluciones mediante el protocolo y '
            'estándar de la empresa. Tecnologías empleadas: Java 1.8, Struts, '
            'Springboot, Maven, DB2, IBM Websphere, Selenium, Jprofiler, IBM '
            'MQ, Jenkins y SVN. Instituto Nicaragüense de Seguridad Social '
            'Backend Developer 2014 2019 Desarrollo e implementación de '
            'sistemas, así como de nuevas características en los sistemas '
            'existentes en la institución. Instalación y configuración de '
            'herramientas devops. Creacion y administración de base de datos. '
            'Tecnologías empleadas: Java 1.8, ExtJs, Spring, Maven, Hibernate,'
            ' JSF Primefaces , PostgreSQL, Jenkins, Git, Springboot. Agility'
            ' First Web Developer 2013 2014 Desarrollo de nuevas '
            'características para sistema de certificación de tutores y '
            'maestros para niños y adolescentes en USA. Tecnologías empleadas:'
            ' Ruby, Ruby on Rails, Mysql, Git, Vagrant, Chef, Puppet, '
            'Bootstrap. Instituto de Ciencias Sostenibles Backend Developer'
            ' 2012 2013 Análisis, desarrollo e implementación de sistemas '
            'para el área de la salud MINSA . Tecnologías empleadas: Java 1.6,'
            ' Hibernate, Maven, JSF Primefaces , Oracle XE, Mysql. ABOUT Soy '
            'Franciso, graduado de Ingeniería en Computación con experiencia '
            'laboral de 8+ años en Java. El aprendizaje ha sido constante, '
            'consiguiendo varias habilidades y aptitudes que me permiten '
            'desenvolverme en el area de trabajo. Capacidad de liderazgo y '
            'motivado a conseguir el éxito personal y profesional. SKILLS '
            'Java SQL PostgreSQL Jenkins Ruby Ruby on Rails RESUME'
        )
    },
    {
        'name': 'job_description',
        'resume': (
            'Java Developer'
            "You bring to Applaudo the following competencies: Bachelor's"
            ' Degree or higher in Computer Science or Computer Engineering '
            'or related field or equivalent experience. 2+ years of '
            'professional software development. 2+ years of professional'
            ' experience in Java. Kubernetes optional Pub / Sub models, '
            'ideally Kafka optional Excellent troubleshooting skills.'
            'Strong technical leadership and guidance skills.'
            'Outstanding skills at interacting with people, both within the'
            ' organization from developers to senior management and with '
            'customers/partners. Responsible, organized, and hardworking '
            'with excellent communication skills. English is a requirement,'
            ' as you will be working directly with US based clients.'
            'You will be accountable for the following responsibilities: Code'
            ' custom, applications, from scratch, with minimal supervision. '
            'Define server API requirements, develop RESTful web services, '
            'and process the resulting JSON or XML. Work with Product'
            ' Management to take detailed story driven requirements and '
            'implement them using Agile Test Driven techniques. Work in '
            'an organized team oriented environment with shared '
            'responsibilities. Develop documentation throughout the SDLC.'
        )
    }
]


def build_dataframe(candidates_info: List, clean_complete=True) -> pd.DataFrame:
    candidates_df = pd.DataFrame.from_dict(candidates_info)

    candidates_df['norm-content'] = candidates_df['resume'].apply(
        lambda x: normalizeText(x) 
        )
    candidates_df['clean-content'] = candidates_df['norm-content'].apply(
        lambda x: cleanText(x, clean_complete)
        )
    candidates_df['senten-content'] = candidates_df['clean-content'].apply(
        lambda x: sentenceSegmentation(x)
        )
    return candidates_df


def normalizeText(raw_text: str) -> str:
    return unicodedata.normalize('NFKD', raw_text) 


def cleanText(norm_text: str, complete=True) -> str:
    text = norm_text.replace('\n', " ")
    text = text.replace('\t', " ")
    if complete:
        text = re.sub((
            r"\S+@\S+|"
            r"[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}"
            + r"\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)|"
            r"<.\w+>|"
            r"\d{5,}|"
            r"\w{20,}|"
            r"[^a-zA-Z0-9 .,:;\\]{4,}|"
            r"[\$\%\(\)\*\-\`\|\~\§\©\«\®\°\»\ı\β\–\—\‘\’\“\”]"),
            " ",
            text
        )
        text = re.sub(
            r"[⁄⇢−■▪○●◦★✓✔❖➔➢⬡•·]",
            ".",
            text
        )
        text = re.sub(r"\s{2,}", " ", text)
    else:
        text = re.sub(r"\s{2,}", " ", text)

    return text


def sentenceSegmentation(clean_text: str) -> List[str]:
    # nlp = spacy.load("xx_sent_ud_sm")
    nlp = spacy.load(SPACY_MODEL_XX_PATH)
    doc = nlp(clean_text)
    return [sent.text for sent in doc.sents]


def separating_idx_sentences_list(
    candidate_sentences_row: pd.Series
    ) -> List[List[Tuple]]:

    candidate_idx_sentences = [
        (candidate_sentences_row.name, sentence) 
        for sentence in candidate_sentences_row['senten-content']
        ]

    return candidate_idx_sentences


def creating_idx_sentence_one_complete(
    candidates_idx_sentences: List[List[Tuple]]
    ) -> List[Tuple]:

    candidate_idx_sentences_one_tuplelist = [
        (idx, sentence)
        for idx_sentences in candidates_idx_sentences
        for idx, sentence in idx_sentences
        ]
    
    return candidate_idx_sentences_one_tuplelist


def creating_only_sentences_list(
    candidates_idx_sentences: List[Tuple]
    ) -> List[str]:
    
    candidates_sentences_only_complete = [
        sentence[1]
        for sentence in candidates_idx_sentences
        ]
    
    return candidates_sentences_only_complete


def build_idx_sentences_list(
    candidates_df: pd.DataFrame) -> Tuple[List[Tuple], List[str]]:
    
    candidates_idx_sentences_list = candidates_df.apply(
        separating_idx_sentences_list, axis=1
        )
    
    candidate_idx_sentences_one_tuplelist = (
        creating_idx_sentence_one_complete(candidates_idx_sentences_list)
    )

    candidates_sentences_one_list = (
        creating_only_sentences_list(candidate_idx_sentences_one_tuplelist)
    )

    return candidate_idx_sentences_one_tuplelist, candidates_sentences_one_list


def sort_sentences_by_length(sentences_list: List[str]) -> List[str]:
    length_sorted_idx = np.argsort(
        [-_text_length(sen) for sen in sentences_list]
        )
    sentences_sorted = [sentences_list[idx] for idx in length_sorted_idx]

    return sentences_sorted


def _text_length(text):
  """

  taken from the sentence-transformer repo

  Help function to get the length for the input text. Text can be either
  a list of ints (which means a single text as input), or a tuple of list of ints
  (representing several text inputs to the model).
  """

  if isinstance(text, dict):              #{key: value} case
    return len(next(iter(text.values())))
  elif not hasattr(text, '__len__'):      #Object has no len() method
    return 1
  elif len(text) == 0 or isinstance(text[0], int):    #Empty string or list of ints
    return len(text)
  else:
    return sum([len(t) for t in text])      #Sum of length of individual strings


def cosine_similarity_sentences(embeddings_candidates, embeddings_jd):
     cos_sim_data = cosine_similarity(embeddings_candidates, embeddings_jd)
     print(cos_sim_data.shape)

     return cos_sim_data
    

def building_df_from_tuple(list_of_tuples, columns_names) -> pd.DataFrame:
    group_df = pd.DataFrame(
        list_of_tuples, 
        columns = columns_names
        )
    return group_df


def reducing_cv_sim_from_groups(
    group_df, group_column, reduce_column) -> pd.DataFrame:
    
    groupby_cv = group_df.groupby(group_column)
    sim_per_group = groupby_cv[reduce_column].agg(reducing_groups_similarities)

    return sim_per_group.tolist()

def reducing_groups_similarities(group) -> float:
    complete_sim_array = np.stack(np.array(group.values))
    max_sim_per_senten = complete_sim_array.max(axis=0)

    return max_sim_per_senten.mean()
