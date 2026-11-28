# Part III: Explore Generated Outputs

## Example 1 (Coordinated object)

The Bi-LSTM correctly translates both animals but uses the dative plural form *Kühen* instead of the accusative 
plural form *Kühe*. Since sheep and cows form a coordinated direct object of the verb *füttert*, both nouns 
require the accusative case. This is a grammatical error, although the original meaning is preserved.

The Transformer makes two errors. It uses *sein* instead of *seine*, failing to account for the plural noun, 
and repeats *Schafe* instead of translating cows as *Kühe*. Unlike the Bi-LSTM, the Transformer introduces 
both grammatical and semantic errors.

One possible explanation for the Bi-LSTM error is that its sequential decoder struggles to maintain the 
grammatical relationship between the verb and the second coordinated noun. The Transformer, on the other 
hand, may rely too heavily on the previously generated word *Schafe* when predicting the next noun, 
resulting in repetition. Greedy decoding may also contribute to this error by always selecting the most 
probable next token.

The limited size and linguistic diversity of the Multi30k dataset may contribute to both models' 
difficulties in learning grammatical agreement and coordinated noun structures.



## Example 2 (Coordinated subject)

Both the Bi-LSTM and Transformer produce translations identical to the Gold reference, with no 
identifiable grammatical or semantic errors. Both correctly translate the coordinated 
subject *Zwei Jungen und Mädchen* and use the plural verb form *sitzen*, maintaining subject-verb 
agreement. They also preserve the ambiguity of the original sentence, where the numeral "two" may 
refer only to the boys or to both groups.

Unlike Example 1, where the coordinated nouns form an accusative object, the coordinated nouns 
here form a nominative subject. The absence of a possessive also reduces the grammatical 
complexity of the construction.

The Bi-LSTM's bidirectional encoder and global attention mechanism may help it capture the 
relationship between the coordinated nouns, while the Transformer's self-attention mechanism 
allows it to directly model relationships between the subject and verb. Additionally, similar 
descriptions of people performing activities may occur frequently in the Multi30k image-caption 
dataset, helping both models learn this sentence structure.


## Example 3 (Directional adverb)

The Bi-LSTM produces a translation identical to the Gold reference, with no identifiable grammatical 
or semantic errors. The Transformer uses *blicken* instead of *schauen*, which is an acceptable 
synonym and therefore not a translation error. However, it generates *nach unten* (downwards) 
instead of *nach oben* (upwards). Although the sentence remains grammatically correct, this 
semantic error reverses the intended direction.

The Bi-LSTM's bidirectional encoder and global attention mechanism may help it capture the relationship 
between the verb and directional adverb in the source sentence. In contrast, the Transformer may rely 
too heavily on learned target-language patterns rather than the specific directional information in the 
source. Greedy decoding may also contribute to this error by selecting the most probable token even when 
an alternative would produce a more accurate translation.

Since opposite directional expressions can appear in similar contexts within the Multi30k dataset, 
differences in their frequency may also influence the predictions. Furthermore, the use of *blicken* 
instead of *schauen* demonstrates a limitation of BLEU, as valid synonyms may be penalised despite 
preserving the intended meaning.


## Example 4 (Verb selection and subject-verb agreement)

The Bi-LSTM produces a grammatically and semantically correct translation, using *Lkw*, an 
abbreviation of *Lastkraftwagen*. It also correctly translates the English present 
progressive "is loading" using the German simple present form *lädt*. The difference in noun 
representation demonstrates another limitation of BLEU, as equivalent expressions may be penalised.

The Transformer makes two errors. Firstly, it uses *betreten* (to enter) instead of 
*lädt* (loads), changing the meaning of the action. Secondly, it uses a plural verb form 
despite the grammatical head of the subject, *Gruppe*, being singular. Even if entering were 
the intended action, the appropriate verb form would be *betritt*.

The Bi-LSTM's bidirectional encoder and global attention mechanism may help it preserve the 
source action and grammatical relationships. In contrast, the Transformer may incorrectly 
associate the verb with the nearby plural noun *Männern* rather than the singular noun *Gruppe*, 
exhibiting possible agreement attraction. Although self-attention can capture dependencies 
between tokens, it does not explicitly enforce grammatical rules.

The limited linguistic diversity of Multi30k and potential underrepresentation of collective 
noun constructions may also contribute to these errors. Furthermore, similar image-caption contexts 
involving vehicles and different activities may influence the model's verb selection.


## Example 5 (Subject-verb agreement)

The Bi-LSTM produces a translation identical to the Gold reference, with no identifiable grammatical or 
semantic errors. The Transformer uses *balanciert* instead of *balancieren*, failing to maintain third-person 
plural agreement with the subject *Zwei Kinder*. This is a grammatical error, although the intended 
meaning remains understandable.

Interestingly, the Transformer exhibits the opposite agreement error to Example 4. In Example 4, 
it generates a plural verb for a singular subject, whereas here it generates a singular verb form 
for a plural subject. Since there are no intervening nouns between the subject and verb, this error 
suggests inconsistent learning of grammatical agreement rather than solely difficulty with long-distance 
dependencies.

The Bi-LSTM's recurrent decoder may successfully retain information about the plural subject when 
generating the verb. Although the Transformer's self-attention mechanism allows direct access to the 
subject, it does not explicitly enforce grammatical agreement. Potential imbalances in the frequency 
of singular and plural verb forms, alongside the limited size and linguistic diversity of Multi30k, 
may also contribute to this error.


## Part III - Overall Findings

### Translation Performance

| Metric | Bi-LSTM | Transformer |
|---|---:|---:|
| Error-free translations | 4/5 | 1/5 |
| Translations containing errors | 1/5 | 4/5 |

### Error Analysis Summary

| Example | Linguistic Phenomenon | Bi-LSTM Errors | Transformer Errors |
|---|---|---|---|
| 1 | Coordinated object | Grammatical | Grammatical and semantic |
| 2 | Coordinated subject | None | None |
| 3 | Directional adverb | None | Semantic |
| 4 | Verb selection and agreement | None | Grammatical and semantic |
| 5 | Subject-verb agreement | None | Grammatical |

### Error Distribution

| Error Category | Bi-LSTM | Transformer |
|---|---:|---:|
| Grammatical errors only | 1 | 1 |
| Semantic errors only | 0 | 1 |
| Both grammatical and semantic errors | 0 | 2 |
| No errors | 4 | 1 |

### Key Findings

- The Bi-LSTM outperformed the Transformer across the five provided examples.
- Both models successfully translated the coordinated subject in Example 2 but struggled with the coordinated object in Example 1.
- The Transformer exhibited inconsistent grammatical agreement, incorrect lexical selection and word repetition.
- Grammatical errors do not necessarily prevent the intended meaning from being understood.
- Grammatically correct translations can still contain significant semantic errors.
- Valid synonyms and abbreviations demonstrate the limitations of BLEU as an evaluation metric.
- Architectural differences, limited training data and potential dataset biases may contribute to the observed differences in performance.

These conclusions are based on the five provided examples only and cannot be generalised to the overall translation performance of either model. 