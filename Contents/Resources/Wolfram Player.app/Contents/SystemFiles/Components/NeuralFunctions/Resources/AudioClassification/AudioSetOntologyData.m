(* Created with the Wolfram Language : www.wolfram.com *)
<|"Data" -> <|"Human sounds" -> <|"DepthIndex" -> 0., 
     "FirstChildren" -> {"Human voice", "Whistling", "Respiratory sounds", 
       "Human locomotion", "Digestive", "Hands", "Heart sounds, heartbeat", 
       "Otoacoustic emission", "Human group actions"}, "FirstParents" -> {}, 
     "FlattenedChildren" -> {"Human voice", "Whistling", 
       "Respiratory sounds", "Human locomotion", "Digestive", "Hands", 
       "Heart sounds, heartbeat", "Otoacoustic emission", 
       "Human group actions", "Speech", "Shout", "Screaming", "Whispering", 
       "Laughter", "Crying, sobbing", "Wail, moan", "Sigh", "Singing", 
       "Humming", "Groan", "Grunt", "Yawn", "Wolf-whistling", "Breathing", 
       "Cough", "Sneeze", "Sniff", "Run", "Shuffle", "Walk, footsteps", 
       "Chewing, mastication", "Biting", "Gargling", "Stomach rumble", 
       "Burping, eructation", "Hiccup", "Fart", "Finger snapping", 
       "Clapping", "Heart murmur", "Tinnitus, ringing in the ears", 
       "Children shouting", "Clapping", "Cheering", "Applause", "Chatter", 
       "Crowd", "Hubbub, speech noise, speech babble", "Booing", 
       "Children playing", "Male speech, man speaking", 
       "Female speech, woman speaking", "Child speech, kid speaking", 
       "Conversation", "Narration, monologue", "Babbling", 
       "Speech synthesizer", "Bellow", "Whoop", "Yell", "Battle cry", 
       "Children shouting", "Baby laughter", "Giggle", "Snicker", 
       "Belly laugh", "Chuckle, chortle", "Baby cry, infant cry", "Whimper", 
       "Choir", "Yodeling", "Chant", "Male singing", "Female singing", 
       "Child singing", "Synthetic singing", "Rapping", "Wheeze", "Snoring", 
       "Gasp", "Pant", "Snort", "Throat clearing", "Mantra"}, 
     "FlattenedParents" -> {}, "BottomDepth" -> 4, "TopDepth" -> 0, 
     "Restrictions" -> {"abstract"}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/0dgw9r", "EntityCanonicalName" -> 
      "HumanSounds::573y7"|>, "Human voice" -> <|"DepthIndex" -> 0.25, 
     "FirstChildren" -> {"Speech", "Shout", "Screaming", "Whispering", 
       "Laughter", "Crying, sobbing", "Wail, moan", "Sigh", "Singing", 
       "Humming", "Groan", "Grunt", "Yawn"}, "FirstParents" -> 
      {"Human sounds"}, "FlattenedChildren" -> {"Speech", "Shout", 
       "Screaming", "Whispering", "Laughter", "Crying, sobbing", 
       "Wail, moan", "Sigh", "Singing", "Humming", "Groan", "Grunt", "Yawn", 
       "Male speech, man speaking", "Female speech, woman speaking", 
       "Child speech, kid speaking", "Conversation", "Narration, monologue", 
       "Babbling", "Speech synthesizer", "Bellow", "Whoop", "Yell", 
       "Battle cry", "Children shouting", "Baby laughter", "Giggle", 
       "Snicker", "Belly laugh", "Chuckle, chortle", "Baby cry, infant cry", 
       "Whimper", "Choir", "Yodeling", "Chant", "Male singing", 
       "Female singing", "Child singing", "Synthetic singing", "Rapping", 
       "Mantra"}, "FlattenedParents" -> {"Human sounds"}, "BottomDepth" -> 3, 
     "TopDepth" -> 1, "Restrictions" -> {"abstract"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/09l8g", "EntityCanonicalName" -> 
      "HumanVoice::4397v"|>, "Speech" -> 
    <|"DepthIndex" -> 0.6666666666666666, "FirstChildren" -> 
      {"Male speech, man speaking", "Female speech, woman speaking", 
       "Child speech, kid speaking", "Conversation", "Narration, monologue", 
       "Babbling", "Speech synthesizer"}, "FirstParents" -> {"Human voice"}, 
     "FlattenedChildren" -> {"Male speech, man speaking", 
       "Female speech, woman speaking", "Child speech, kid speaking", 
       "Conversation", "Narration, monologue", "Babbling", 
       "Speech synthesizer"}, "FlattenedParents" -> 
      {"Human voice", "Human sounds"}, "BottomDepth" -> 1, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.48829170666788574, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(11.36892030661717 - 13.586599180081805*x))^(-1)], 
     "AudioSetID" -> "/m/09x0r", "EntityCanonicalName" -> "Speech::8x82c"|>, 
   "Male speech, man speaking" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Speech"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Speech", "Human voice", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.008555943292240017, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(5.66608089740441 - 9.798628628834814*x))^(-1)], 
     "AudioSetID" -> "/m/05zppz", "EntityCanonicalName" -> 
      "MaleSpeechManSpeaking::db2gf"|>, "Female speech, woman speaking" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Speech"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Speech", "Human voice", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.004082410130594456, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.423041114736269 - 10.170545588256816*x))^(-1)], 
     "AudioSetID" -> "/m/02zsn", "EntityCanonicalName" -> 
      "FemaleSpeechWomanSpeaking::fqhn3"|>, "Child speech, kid speaking" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Speech"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Speech", "Human voice", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.005432546515703336, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.45908429460593 - 8.913946425220253*x))^(-1)], 
     "AudioSetID" -> "/m/0ytgt", "EntityCanonicalName" -> 
      "ChildSpeechKidSpeaking::ws3z4"|>, "Conversation" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Speech"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Speech", "Human voice", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0010448572400936662, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.533965807375058 - 77.05475821078753*x))^(-1)], 
     "AudioSetID" -> "/m/01h8n0", "EntityCanonicalName" -> 
      "Conversation::kdv2y"|>, "Narration, monologue" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Speech"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Speech", "Human voice", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.007517689427995266, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(5.716622574395801 - 10.52907584927406*x))^(-1)], 
     "AudioSetID" -> "/m/02qldy", "EntityCanonicalName" -> 
      "NarrationMonologue::wkm6j"|>, "Babbling" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Speech"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Speech", "Human voice", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0003626777197019338, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.75744195186994 - 25.075295970562784*x))^(-1)], 
     "AudioSetID" -> "/m/0261r1", "EntityCanonicalName" -> 
      "Babbling::7v86g"|>, "Speech synthesizer" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Speech"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Speech", "Human voice", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0007700552143811367, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.843337731829406 - 26.364168107694418*x))^(-1)], 
     "AudioSetID" -> "/m/0brhx", "EntityCanonicalName" -> 
      "SpeechSynthesizer::fzc6h"|>, 
   "Shout" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Bellow", "Whoop", "Yell", "Battle cry", 
       "Children shouting"}, "FirstParents" -> {"Human voice"}, 
     "FlattenedChildren" -> {"Bellow", "Whoop", "Yell", "Battle cry", 
       "Children shouting"}, "FlattenedParents" -> 
      {"Human voice", "Human sounds"}, "BottomDepth" -> 1, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0006227491377514997, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.92216135914815 - 22.336540752852986*x))^(-1)], 
     "AudioSetID" -> "/m/07p6fty", "EntityCanonicalName" -> "Shout::5xgdr"|>, 
   "Bellow" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Shout"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Shout", "Human voice", "Human sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00019302175558366224, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.961232098791074 - 167.04944887039557*x))^(-1)], 
     "AudioSetID" -> "/m/07q4ntr", "EntityCanonicalName" -> 
      "Bellow::5bdkn"|>, "Whoop" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Shout"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Shout", "Human voice", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0008792648918824193, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.678217493940259 - 62.58583384866426*x))^(-1)], 
     "AudioSetID" -> "/m/07rwj3x", "EntityCanonicalName" -> "Whoop::33kh3"|>, 
   "Yell" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Shout"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Shout", "Human voice", "Human sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00033474036034114054, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.20201937164122 - 47.24624160021935*x))^(-1)], 
     "AudioSetID" -> "/m/07sr1lc", "EntityCanonicalName" -> "Yell::7j975"|>, 
   "Battle cry" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Shout"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Shout", "Human voice", "Human sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00016305258826935677, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.347569635559497 - 36.63544845380466*x))^(-1)], 
     "AudioSetID" -> "/m/04gy_2", "EntityCanonicalName" -> 
      "BattleCry::6p765"|>, "Children shouting" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Shout", "Human group actions"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Shout", "Human group actions", "Human voice", 
       "Human sounds", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> <|"Shout" -> 3, "Human group actions" -> 2|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0002783576896311761, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.82424575329285 - 31.8560725054264*x))^(-1)], 
     "AudioSetID" -> "/t/dd00135", "EntityCanonicalName" -> 
      "ChildrenShouting::49nkw"|>, "Screaming" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Human voice"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Human voice", "Human sounds"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0005409688676226323, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.103128209807702 - 28.872766673227765*x))^(-1)], 
     "AudioSetID" -> "/m/03qc9zr", "EntityCanonicalName" -> 
      "Screaming::fc8qk"|>, "Whispering" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Human voice"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Human voice", "Human sounds"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0007126566396944161, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.039353161707142 - 15.714259159940351*x))^(-1)], 
     "AudioSetID" -> "/m/02rtxlg", "EntityCanonicalName" -> 
      "Whispering::472q7"|>, "Laughter" -> 
    <|"DepthIndex" -> 0.6666666666666666, "FirstChildren" -> 
      {"Baby laughter", "Giggle", "Snicker", "Belly laugh", 
       "Chuckle, chortle"}, "FirstParents" -> {"Human voice"}, 
     "FlattenedChildren" -> {"Baby laughter", "Giggle", "Snicker", 
       "Belly laugh", "Chuckle, chortle"}, "FlattenedParents" -> 
      {"Human voice", "Human sounds"}, "BottomDepth" -> 1, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.002612397076228355, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.714494669960254 - 11.652374918080737*x))^(-1)], 
     "AudioSetID" -> "/m/01j3sz", "EntityCanonicalName" -> 
      "Laughter::9tb75"|>, "Baby laughter" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Laughter"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Laughter", "Human voice", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0003596300077716654, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.689325719248894 - 20.787568618905684*x))^(-1)], 
     "AudioSetID" -> "/t/dd00001", "EntityCanonicalName" -> 
      "BabyLaughter::v3647"|>, "Giggle" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Laughter"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Laughter", "Human voice", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0004210921983654105, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.416137347102701 - 24.923196001858752*x))^(-1)], 
     "AudioSetID" -> "/m/07r660_", "EntityCanonicalName" -> 
      "Giggle::tpr3t"|>, "Snicker" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Laughter"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Laughter", "Human voice", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0008365969248586623, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.388801297495832 - 20.303438584398098*x))^(-1)], 
     "AudioSetID" -> "/m/07s04w4", "EntityCanonicalName" -> 
      "Snicker::2vgk5"|>, "Belly laugh" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Laughter"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Laughter", "Human voice", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0003479471120389701, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.574623721945848 - 25.51891098936937*x))^(-1)], 
     "AudioSetID" -> "/m/07sq110", "EntityCanonicalName" -> 
      "BellyLaugh::4fytp"|>, "Chuckle, chortle" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Laughter"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Laughter", "Human voice", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0007609120785903316, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.771944178680188 - 36.085307883500064*x))^(-1)], 
     "AudioSetID" -> "/m/07rgt08", "EntityCanonicalName" -> 
      "ChuckleChortle::n3782"|>, "Crying, sobbing" -> 
    <|"DepthIndex" -> 0.6666666666666666, "FirstChildren" -> 
      {"Baby cry, infant cry", "Whimper"}, "FirstParents" -> {"Human voice"}, 
     "FlattenedChildren" -> {"Baby cry, infant cry", "Whimper"}, 
     "FlattenedParents" -> {"Human voice", "Human sounds"}, 
     "BottomDepth" -> 1, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0006384956493912195, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.071240546272405 - 21.44375394131692*x))^(-1)], 
     "AudioSetID" -> "/m/0463cq4", "EntityCanonicalName" -> 
      "CryingSobbing::nt8mj"|>, "Baby cry, infant cry" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Crying, sobbing"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Crying, sobbing", "Human voice", 
       "Human sounds"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0010956524389314722, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.60382577383636 - 12.78168114087736*x))^(-1)], 
     "AudioSetID" -> "/t/dd00002", "EntityCanonicalName" -> 
      "BabyCryInfantCry::5f2d3"|>, "Whimper" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Crying, sobbing"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Crying, sobbing", "Human voice", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0005496040514250593, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.717286373283736 - 25.34350521229597*x))^(-1)], 
     "AudioSetID" -> "/m/07qz6j3", "EntityCanonicalName" -> 
      "Whimper::44mx2"|>, "Wail, moan" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Human voice"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Human voice", "Human sounds"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00004520772696564721, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(11.639668731229346 - 783.5997667132299*x))^(-1)], 
     "AudioSetID" -> "/m/07qw_06", "EntityCanonicalName" -> 
      "WailMoan::2f7nr"|>, "Sigh" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Human voice"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Human voice", "Human sounds"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00009447906983831889, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(10.049653445440933 - 329.5791253366145*x))^(-1)], 
     "AudioSetID" -> "/m/07plz5l", "EntityCanonicalName" -> "Sigh::yp622"|>, 
   "Singing" -> <|"DepthIndex" -> 0.5, "FirstChildren" -> 
      {"Choir", "Yodeling", "Chant", "Male singing", "Female singing", 
       "Child singing", "Synthetic singing", "Rapping"}, 
     "FirstParents" -> {"Human voice"}, "FlattenedChildren" -> 
      {"Choir", "Yodeling", "Chant", "Male singing", "Female singing", 
       "Child singing", "Synthetic singing", "Rapping", "Mantra"}, 
     "FlattenedParents" -> {"Human voice", "Human sounds"}, 
     "BottomDepth" -> 2, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.020363287262087987, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(5.518467979759072 - 9.888977528782952*x))^(-1)], 
     "AudioSetID" -> "/m/015lz1", "EntityCanonicalName" -> 
      "Singing::n4c3j"|>, "Choir" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> 
      {"Singing", "Musical instrument"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Singing", "Musical instrument", "Human voice", 
       "Music", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> <|"Singing" -> 3, "Musical instrument" -> 2|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.003275782373050099, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.175686534187394 - 10.977664046205426*x))^(-1)], 
     "AudioSetID" -> "/m/0l14jd", "EntityCanonicalName" -> "Choir::rnbz4"|>, 
   "Yodeling" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Singing"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Singing", "Human voice", "Human sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.000182354763827723, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.805562244428225 - 79.27413096843259*x))^(-1)], 
     "AudioSetID" -> "/m/01swy6", "EntityCanonicalName" -> 
      "Yodeling::v32q3"|>, "Chant" -> <|"DepthIndex" -> 0.75, 
     "FirstChildren" -> {"Mantra"}, "FirstParents" -> 
      {"Singing", "Vocal music"}, "FlattenedChildren" -> {"Mantra"}, 
     "FlattenedParents" -> {"Singing", "Vocal music", "Human voice", 
       "Music genre", "Human sounds", "Music"}, "BottomDepth" -> 1, 
     "TopDepth" -> <|"Singing" -> 3, "Vocal music" -> 3|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0007847858220441004, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.026349680987485 - 22.424545810576486*x))^(-1)], 
     "AudioSetID" -> "/m/02bk07", "EntityCanonicalName" -> "Chant::qw4h3"|>, 
   "Mantra" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Chant"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Chant", "Singing", "Vocal music", "Human voice", 
       "Music genre", "Human sounds", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 4, "Restrictions" -> {}, "ClassPrior" -> 
      0.0023472461382950086, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.046631787198664 - 9.541322755263716*x))^(-1)], 
     "AudioSetID" -> "/m/01c194", "EntityCanonicalName" -> "Mantra::gb98d"|>, 
   "Male singing" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Singing"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Singing", "Human voice", "Human sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.003507408479750494, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.5653937225425 - 16.04511634677385*x))^(-1)], 
     "AudioSetID" -> "/t/dd00003", "EntityCanonicalName" -> 
      "MaleSinging::p47f2"|>, "Female singing" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Singing"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Singing", "Human voice", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0038619589676383787, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.735970166302857 - 11.046150595941702*x))^(-1)], 
     "AudioSetID" -> "/t/dd00004", "EntityCanonicalName" -> 
      "FemaleSinging::wt9jz"|>, "Child singing" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Singing"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Singing", "Human voice", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0012581970752124508, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.642120278600607 - 12.888342676454037*x))^(-1)], 
     "AudioSetID" -> "/t/dd00005", "EntityCanonicalName" -> 
      "ChildSinging::wtzh6"|>, "Synthetic singing" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Singing"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Singing", "Human voice", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0002595634660611879, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.874676263255305 - 19.834917085741125*x))^(-1)], 
     "AudioSetID" -> "/t/dd00006", "EntityCanonicalName" -> 
      "SyntheticSinging::8hmw7"|>, "Rapping" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Singing"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Singing", "Human voice", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.002124763167385419, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.27488819791077 - 15.05877163117641*x))^(-1)], 
     "AudioSetID" -> "/m/06bxc", "EntityCanonicalName" -> "Rapping::494x6"|>, 
   "Humming" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Human voice"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Human voice", "Human sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0001493378845831492, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.95061014055186 - 50.31356067982524*x))^(-1)], 
     "AudioSetID" -> "/m/02fxyj", "EntityCanonicalName" -> 
      "Humming::8yjkg"|>, "Groan" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Human voice"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Human voice", "Human sounds"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00028191335354982246, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.91079765352564 - 41.11863051000315*x))^(-1)], 
     "AudioSetID" -> "/m/07s2xch", "EntityCanonicalName" -> "Groan::84z9d"|>, 
   "Grunt" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Human voice"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Human voice", "Human sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00010260630165236781, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.47139093583241 - 35.4059949431338*x))^(-1)], 
     "AudioSetID" -> "/m/07r4k75", "EntityCanonicalName" -> "Grunt::727vj"|>, 
   "Yawn" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Human voice"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Human voice", "Human sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/01j423", "EntityCanonicalName" -> "Yawn::936xk"|>, 
   "Whistling" -> <|"DepthIndex" -> 0.5, "FirstChildren" -> 
      {"Wolf-whistling"}, "FirstParents" -> {"Human sounds"}, 
     "FlattenedChildren" -> {"Wolf-whistling"}, "FlattenedParents" -> 
      {"Human sounds"}, "BottomDepth" -> 1, "TopDepth" -> 1, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0009092340591967248, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.321455568638408 - 11.729979110657656*x))^(-1)], 
     "AudioSetID" -> "/m/01w250", "EntityCanonicalName" -> 
      "Whistling::jyc9y"|>, "Wolf-whistling" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Whistling"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Whistling", "Human sounds"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/079vc8", "EntityCanonicalName" -> 
      "Wolf-whistling::f9532"|>, "Respiratory sounds" -> 
    <|"DepthIndex" -> 0.3333333333333333, "FirstChildren" -> 
      {"Breathing", "Cough", "Sneeze", "Sniff"}, "FirstParents" -> 
      {"Human sounds"}, "FlattenedChildren" -> {"Breathing", "Cough", 
       "Sneeze", "Sniff", "Wheeze", "Snoring", "Gasp", "Pant", "Snort", 
       "Throat clearing"}, "FlattenedParents" -> {"Human sounds"}, 
     "BottomDepth" -> 2, "TopDepth" -> 1, "Restrictions" -> {"abstract"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/09hlz4", "EntityCanonicalName" -> 
      "RespiratorySounds::89k53"|>, "Breathing" -> 
    <|"DepthIndex" -> 0.6666666666666666, "FirstChildren" -> 
      {"Wheeze", "Snoring", "Gasp", "Pant", "Snort"}, 
     "FirstParents" -> {"Respiratory sounds"}, "FlattenedChildren" -> 
      {"Wheeze", "Snoring", "Gasp", "Pant", "Snort"}, 
     "FlattenedParents" -> {"Respiratory sounds", "Human sounds"}, 
     "BottomDepth" -> 1, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0003403278322132992, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.203851151888616 - 61.28094732002594*x))^(-1)], 
     "AudioSetID" -> "/m/0lyf6", "EntityCanonicalName" -> 
      "Breathing::wb4wk"|>, "Wheeze" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Breathing"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Breathing", "Respiratory sounds", "Human sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00003352483123295186, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(10.951203711699403 - 550.9935234800574*x))^(-1)], 
     "AudioSetID" -> "/m/07mzm6", "EntityCanonicalName" -> "Wheeze::5h4rp"|>, 
   "Snoring" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Breathing"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Breathing", "Respiratory sounds", 
       "Human sounds"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0010890490630825575, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.146285876285376 - 9.94731213255536*x))^(-1)], 
     "AudioSetID" -> "/m/01d3sd", "EntityCanonicalName" -> 
      "Snoring::j9949"|>, "Gasp" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Breathing"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Breathing", "Respiratory sounds", "Human sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0000980347337569653, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(9.309481115164305 - 144.98560113720913*x))^(-1)], 
     "AudioSetID" -> "/m/07s0dtb", "EntityCanonicalName" -> "Gasp::3w462"|>, 
   "Pant" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Breathing"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Breathing", "Respiratory sounds", 
       "Human sounds"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0000807643661521113, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.815461844864366 - 39.748388994113704*x))^(-1)], 
     "AudioSetID" -> "/m/07pyy8b", "EntityCanonicalName" -> "Pant::fcng5"|>, 
   "Snort" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Breathing"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Breathing", "Respiratory sounds", 
       "Human sounds"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00019302175558366224, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.589633509571517 - 47.40073290191059*x))^(-1)], 
     "AudioSetID" -> "/m/07q0yl5", "EntityCanonicalName" -> "Snort::w3c7t"|>, 
   "Cough" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Throat clearing"}, "FirstParents" -> 
      {"Respiratory sounds"}, "FlattenedChildren" -> {"Throat clearing"}, 
     "FlattenedParents" -> {"Respiratory sounds", "Human sounds"}, 
     "BottomDepth" -> 1, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0003626777197019338, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.242990631278918 - 36.18876163195187*x))^(-1)], 
     "AudioSetID" -> "/m/01b_21", "EntityCanonicalName" -> "Cough::5956r"|>, 
   "Throat clearing" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Cough"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Cough", "Respiratory sounds", "Human sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00011530510136181928, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.103505597713085 - 31.668679872723178*x))^(-1)], 
     "AudioSetID" -> "/m/0dl9sf8", "EntityCanonicalName" -> 
      "ThroatClearing::zs3g4"|>, "Sneeze" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Respiratory sounds"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Respiratory sounds", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0005018565645175218, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.065918515362556 - 20.53350053882292*x))^(-1)], 
     "AudioSetID" -> "/m/01hsr_", "EntityCanonicalName" -> "Sneeze::tsj4g"|>, 
   "Sniff" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Respiratory sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Respiratory sounds", "Human sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0000416520630470008, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(10.446723294843324 - 213.97281381270625*x))^(-1)], 
     "AudioSetID" -> "/m/07ppn3j", "EntityCanonicalName" -> "Sniff::v2g25"|>, 
   "Human locomotion" -> <|"DepthIndex" -> 0.5, "FirstChildren" -> 
      {"Run", "Shuffle", "Walk, footsteps"}, "FirstParents" -> 
      {"Human sounds"}, "FlattenedChildren" -> {"Run", "Shuffle", 
       "Walk, footsteps"}, "FlattenedParents" -> {"Human sounds"}, 
     "BottomDepth" -> 1, "TopDepth" -> 1, "Restrictions" -> {"abstract"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/0bpl036", "EntityCanonicalName" -> 
      "HumanLocomotion::pgh4c"|>, "Run" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Human locomotion"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Human locomotion", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0021384778710716265, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.919897240317306 - 8.236595867385027*x))^(-1)], 
     "AudioSetID" -> "/m/06h7j", "EntityCanonicalName" -> "Run::5725v"|>, 
   "Shuffle" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Human locomotion"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Human locomotion", "Human sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00022095911494445544, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.50275419278237 - 46.333066418319504*x))^(-1)], 
     "AudioSetID" -> "/m/07qv_x_", "EntityCanonicalName" -> 
      "Shuffle::pf993"|>, "Walk, footsteps" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Human locomotion"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Human locomotion", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0007659915984741122, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.442141965699054 - 25.312073326708415*x))^(-1)], 
     "AudioSetID" -> "/m/07pbtc8", "EntityCanonicalName" -> 
      "WalkFootsteps::kz8vf"|>, "Digestive" -> <|"DepthIndex" -> 0.5, 
     "FirstChildren" -> {"Chewing, mastication", "Biting", "Gargling", 
       "Stomach rumble", "Burping, eructation", "Hiccup", "Fart"}, 
     "FirstParents" -> {"Human sounds"}, "FlattenedChildren" -> 
      {"Chewing, mastication", "Biting", "Gargling", "Stomach rumble", 
       "Burping, eructation", "Hiccup", "Fart"}, "FlattenedParents" -> 
      {"Human sounds"}, "BottomDepth" -> 1, "TopDepth" -> 1, 
     "Restrictions" -> {"abstract"}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/0160x5", "EntityCanonicalName" -> 
      "Digestive::443n4"|>, "Chewing, mastication" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Digestive"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Digestive", "Human sounds"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00033016879244573803, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.564242431027637 - 17.26487851741144*x))^(-1)], 
     "AudioSetID" -> "/m/03cczk", "EntityCanonicalName" -> 
      "ChewingMastication::pk7qc"|>, "Biting" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Digestive"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Digestive", "Human sounds"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00011632100533857539, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.1661242617585 - 94.72135477190598*x))^(-1)], 
     "AudioSetID" -> "/m/07pdhp0", "EntityCanonicalName" -> 
      "Biting::2nqz9"|>, "Gargling" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Digestive"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Digestive", "Human sounds"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 7.619279825670877*^-6, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(12.760925346420246 - 37483.127720226534*x))^(-1)], 
     "AudioSetID" -> "/m/0939n_", "EntityCanonicalName" -> 
      "Gargling::4gjpn"|>, "Stomach rumble" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Digestive"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Digestive", "Human sounds"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00006247809457050119, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(10.19583755334288 - 26.41728164439594*x))^(-1)], 
     "AudioSetID" -> "/m/01g90h", "EntityCanonicalName" -> 
      "StomachRumble::2scvk"|>, "Burping, eructation" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Digestive"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Digestive", "Human sounds"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0005765255068090964, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.711396217822537 - 13.646064725605056*x))^(-1)], 
     "AudioSetID" -> "/m/03q5_w", "EntityCanonicalName" -> 
      "BurpingEructation::wv827"|>, "Hiccup" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Digestive"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Digestive", "Human sounds"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0003962025509348856, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.376961782926205 - 62.456719429067846*x))^(-1)], 
     "AudioSetID" -> "/m/02p3nc", "EntityCanonicalName" -> "Hiccup::296m4"|>, 
   "Fart" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Digestive"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Digestive", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0005399529636458762, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.172212891372542 - 12.065948426221723*x))^(-1)], 
     "AudioSetID" -> "/m/02_nn", "EntityCanonicalName" -> "Fart::w4hfp"|>, 
   "Hands" -> <|"DepthIndex" -> 0.5, "FirstChildren" -> 
      {"Finger snapping", "Clapping"}, "FirstParents" -> {"Human sounds"}, 
     "FlattenedChildren" -> {"Finger snapping", "Clapping"}, 
     "FlattenedParents" -> {"Human sounds"}, "BottomDepth" -> 1, 
     "TopDepth" -> 1, "Restrictions" -> {}, "ClassPrior" -> 
      0.00015136969253666145, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(9.483602067055264 - 27.23763286528121*x))^(-1)], 
     "AudioSetID" -> "/m/0k65p", "EntityCanonicalName" -> "Hands::2q394"|>, 
   "Finger snapping" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Hands"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Hands", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0000208260315235004, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(12.365422382625956 - 2308.6684376205835*x))^(-1)], 
     "AudioSetID" -> "/m/025_jnm", "EntityCanonicalName" -> 
      "FingerSnapping::zr7sq"|>, "Clapping" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> 
      {"Hands", "Human group actions"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Hands", "Human group actions", "Human sounds", 
       "Human sounds"}, "BottomDepth" -> 0, "TopDepth" -> 
      <|"Hands" -> 2, "Human group actions" -> 2|>, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00033118469642249416, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.553644668608221 - 19.149441562919183*x))^(-1)], 
     "AudioSetID" -> "/m/0l15bq", "EntityCanonicalName" -> 
      "Clapping::p95y4"|>, "Heart sounds, heartbeat" -> 
    <|"DepthIndex" -> 0.5, "FirstChildren" -> {"Heart murmur"}, 
     "FirstParents" -> {"Human sounds"}, "FlattenedChildren" -> 
      {"Heart murmur"}, "FlattenedParents" -> {"Human sounds"}, 
     "BottomDepth" -> 1, "TopDepth" -> 1, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00044344208585404507, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.252968427658567 - 10.61854929469009*x))^(-1)], 
     "AudioSetID" -> "/m/01jg02", "EntityCanonicalName" -> 
      "HeartSoundsHeartbeat::753h4"|>, "Heart murmur" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Heart sounds, heartbeat"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Heart sounds, heartbeat", "Human sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> 0.00015746511639719814, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.372214573948185 - 13.487732116820894*x))^(-1)], 
     "AudioSetID" -> "/m/01jg1z", "EntityCanonicalName" -> 
      "HeartMurmur::hj427"|>, "Otoacoustic emission" -> 
    <|"DepthIndex" -> 0.5, "FirstChildren" -> 
      {"Tinnitus, ringing in the ears"}, "FirstParents" -> {"Human sounds"}, 
     "FlattenedChildren" -> {"Tinnitus, ringing in the ears"}, 
     "FlattenedParents" -> {"Human sounds"}, "BottomDepth" -> 1, 
     "TopDepth" -> 1, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/04xp5v", "EntityCanonicalName" -> 
      "OtoacousticEmission::s3n6k"|>, "Tinnitus, ringing in the ears" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Otoacoustic emission"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Otoacoustic emission", "Human sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/0pv6y", "EntityCanonicalName" -> 
      "TinnitusRingingInTheEars::f4t54"|>, "Human group actions" -> 
    <|"DepthIndex" -> 0.5, "FirstChildren" -> {"Clapping", "Cheering", 
       "Applause", "Chatter", "Crowd", "Hubbub, speech noise, speech babble", 
       "Booing", "Children playing", "Children shouting"}, 
     "FirstParents" -> {"Human sounds"}, "FlattenedChildren" -> 
      {"Children shouting", "Clapping", "Cheering", "Applause", "Chatter", 
       "Crowd", "Hubbub, speech noise, speech babble", "Booing", 
       "Children playing"}, "FlattenedParents" -> {"Human sounds"}, 
     "BottomDepth" -> 1, "TopDepth" -> 1, "Restrictions" -> {"abstract"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/t/dd00012", "EntityCanonicalName" -> 
      "HumanGroupActions::4fzpv"|>, "Cheering" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Human group actions"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Human group actions", "Human sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0021232393114202844, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.069514927250828 - 27.03255764559258*x))^(-1)], 
     "AudioSetID" -> "/m/053hz1", "EntityCanonicalName" -> 
      "Cheering::63mn3"|>, "Applause" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Human group actions"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Human group actions", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0010458731440704224, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.322608974414024 - 14.27639600297725*x))^(-1)], 
     "AudioSetID" -> "/m/028ght", "EntityCanonicalName" -> 
      "Applause::kz752"|>, "Chatter" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Human group actions"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Human group actions", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0008960273074988952, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.986434100806481 - 16.164141539688742*x))^(-1)], 
     "AudioSetID" -> "/m/07rkbfh", "EntityCanonicalName" -> 
      "Chatter::w8q94"|>, "Crowd" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Human group actions"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Human group actions", "Human sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.005037867820733584, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.409847485412422 - 23.925936030093283*x))^(-1)], 
     "AudioSetID" -> "/m/03qtwd", "EntityCanonicalName" -> "Crowd::y8946"|>, 
   "Hubbub, speech noise, speech babble" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Human group actions", 
       "Noise"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Human group actions", "Noise", "Human sounds", 
       "Channel, environment and background"}, "BottomDepth" -> 0, 
     "TopDepth" -> <|"Human group actions" -> 2, "Noise" -> 2|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0006694807206822811, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.805131639415086 - 30.694633013002*x))^(-1)], 
     "AudioSetID" -> "/m/07qfr4h", "EntityCanonicalName" -> 
      "HubbubSpeechNoiseSpeechBabble::9f66m"|>, 
   "Booing" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Human group actions"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Human group actions", "Human sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/04v5dt", "EntityCanonicalName" -> "Booing::z4n53"|>, 
   "Children playing" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Human group actions"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Human group actions", "Human sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00034185168817843335, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.76758229953705 - 18.418952067133954*x))^(-1)], 
     "AudioSetID" -> "/t/dd00013", "EntityCanonicalName" -> 
      "ChildrenPlaying::zjr2k"|>, "Animal" -> <|"DepthIndex" -> 0., 
     "FirstChildren" -> {"Domestic animals, pets", 
       "Livestock, farm animals, working animals", "Wild animals"}, 
     "FirstParents" -> {}, "FlattenedChildren" -> {"Domestic animals, pets", 
       "Livestock, farm animals, working animals", "Wild animals", "Dog", 
       "Cat", "Horse", "Donkey, ass", "Cattle, bovinae", "Pig", "Goat", 
       "Sheep", "Fowl", "Roaring cats (lions, tigers)", "Bird", 
       "Canidae, dogs, wolves", "Rodents, rats, mice", "Insect", "Frog", 
       "Snake", "Whale vocalization", "Bark", "Yip", "Howl", "Bow-wow", 
       "Growling", "Whimper (dog)", "Bay", "Growling", "Purr", "Meow", 
       "Hiss", "Cat communication", "Caterwaul", "Clip-clop", 
       "Neigh, whinny", "Snort (horse)", "Nicker", "Moo", "Cowbell", "Yak", 
       "Oink", "Bleat", "Bleat", "Chicken, rooster", "Turkey", "Duck", 
       "Goose", "Growling", "Roar", 
       "Bird vocalization, bird call, bird song", "Pigeon, dove", "Crow", 
       "Owl", "Gull, seagull", "Bird flight, flapping wings", "Howl", 
       "Growling", "Mouse", "Chipmunk", "Patter", "Cricket", "Mosquito", 
       "Fly, housefly", "Bee, wasp, etc.", "Croak", "Hiss", "Rattle", 
       "Cluck", "Crowing, cock-a-doodle-doo", "Gobble", "Quack", "Honk", 
       "Chirp, tweet", "Squawk", "Coo", "Caw", "Hoot", "Buzz", "Buzz"}, 
     "FlattenedParents" -> {}, "BottomDepth" -> 4, "TopDepth" -> 0, 
     "Restrictions" -> {}, "ClassPrior" -> 0.01934789123732025, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(5.2147576882613995 - 8.665851248175256*x))^(-1)], 
     "AudioSetID" -> "/m/0jbk", "EntityCanonicalName" -> "Animal::47m82"|>, 
   "Domestic animals, pets" -> <|"DepthIndex" -> 0.3333333333333333, 
     "FirstChildren" -> {"Dog", "Cat"}, "FirstParents" -> {"Animal"}, 
     "FlattenedChildren" -> {"Dog", "Cat", "Bark", "Yip", "Howl", "Bow-wow", 
       "Growling", "Whimper (dog)", "Bay", "Growling", "Purr", "Meow", 
       "Hiss", "Cat communication", "Caterwaul"}, "FlattenedParents" -> 
      {"Animal"}, "BottomDepth" -> 2, "TopDepth" -> 1, "Restrictions" -> {}, 
     "ClassPrior" -> 0.009406762872773266, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(5.888861586127722 - 9.26405497635251*x))^(-1)], 
     "AudioSetID" -> "/m/068hy", "EntityCanonicalName" -> 
      "DomesticAnimalsPets::369zz"|>, 
   "Dog" -> <|"DepthIndex" -> 0.6666666666666666, "FirstChildren" -> 
      {"Bark", "Yip", "Howl", "Bow-wow", "Growling", "Whimper (dog)", "Bay"}, 
     "FirstParents" -> {"Domestic animals, pets"}, "FlattenedChildren" -> 
      {"Bark", "Yip", "Howl", "Bow-wow", "Growling", "Whimper (dog)", "Bay"}, 
     "FlattenedParents" -> {"Domestic animals, pets", "Animal"}, 
     "BottomDepth" -> 1, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.006531246666565077, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.3311942928097755 - 9.17627146939815*x))^(-1)], 
     "AudioSetID" -> "/m/0bt9lr", "EntityCanonicalName" -> "Dog::8p4tb"|>, 
   "Bark" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Dog"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Dog", "Domestic animals, pets", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0012515936993635361, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.169731354093637 - 11.301961514363528*x))^(-1)], 
     "AudioSetID" -> "/m/05tny_", "EntityCanonicalName" -> "Bark::f5qh9"|>, 
   "Yip" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Dog"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Dog", "Domestic animals, pets", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0011058114786990333, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.291102066630884 - 19.533303151789116*x))^(-1)], 
     "AudioSetID" -> "/m/07r_k2n", "EntityCanonicalName" -> "Yip::48bkn"|>, 
   "Howl" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Dog", "Canidae, dogs, wolves"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Dog", "Canidae, dogs, wolves", "Domestic animals, pets", 
       "Wild animals", "Animal", "Animal"}, "BottomDepth" -> 0, 
     "TopDepth" -> <|"Dog" -> 3, "Canidae, dogs, wolves" -> 3|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0003586141037949093, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.030757508461395 - 13.78719168284557*x))^(-1)], 
     "AudioSetID" -> "/m/07qf0zm", "EntityCanonicalName" -> "Howl::6h28d"|>, 
   "Bow-wow" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Dog"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Dog", "Domestic animals, pets", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0018530088536031575, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.996349939197394 - 12.786760361731561*x))^(-1)], 
     "AudioSetID" -> "/m/07rc7d9", "EntityCanonicalName" -> 
      "Bow-wow::6k5j8"|>, "Growling" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Dog", "Cat", 
       "Roaring cats (lions, tigers)", "Canidae, dogs, wolves"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Dog", "Cat", "Roaring cats (lions, tigers)", "Canidae, dogs, wolves", 
       "Domestic animals, pets", "Domestic animals, pets", "Wild animals", 
       "Wild animals", "Animal", "Animal", "Animal", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> <|"Dog" -> 3, "Cat" -> 3, 
       "Roaring cats (lions, tigers)" -> 3, "Canidae, dogs, wolves" -> 3|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0003266131285270916, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.42125494403786 - 17.87847944064264*x))^(-1)], 
     "AudioSetID" -> "/m/0ghcn6", "EntityCanonicalName" -> 
      "Growling::34m97"|>, "Whimper (dog)" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Dog"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Dog", "Domestic animals, pets", "Animal"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0008416764447424429, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.667534280694293 - 21.341432656747806*x))^(-1)], 
     "AudioSetID" -> "/t/dd00136", "EntityCanonicalName" -> 
      "WhimperDog::hkc2r"|>, "Bay" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Dog"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Dog", "Domestic animals, pets", "Animal"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/07srf8z", "EntityCanonicalName" -> "Bay::r2dyp"|>, 
   "Cat" -> <|"DepthIndex" -> 0.6666666666666666, "FirstChildren" -> 
      {"Purr", "Meow", "Hiss", "Cat communication", "Caterwaul", "Growling"}, 
     "FirstParents" -> {"Domestic animals, pets"}, "FlattenedChildren" -> 
      {"Growling", "Purr", "Meow", "Hiss", "Cat communication", "Caterwaul"}, 
     "FlattenedParents" -> {"Domestic animals, pets", "Animal"}, 
     "BottomDepth" -> 1, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0018164363104399373, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.089378595966936 - 8.980664689956596*x))^(-1)], 
     "AudioSetID" -> "/m/01yrx", "EntityCanonicalName" -> "Cat::5yr44"|>, 
   "Purr" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Cat"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Cat", "Domestic animals, pets", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0002067364592698698, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(9.289997631103686 - 10.353863873497545*x))^(-1)], 
     "AudioSetID" -> "/m/02yds9", "EntityCanonicalName" -> "Purr::cv736"|>, 
   "Meow" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Cat"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Cat", "Domestic animals, pets", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0009000909234059196, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.6997103028265785 - 9.480109436871912*x))^(-1)], 
     "AudioSetID" -> "/m/07qrkrw", "EntityCanonicalName" -> "Meow::2hg4c"|>, 
   "Hiss" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Cat", "Snake", "Steam", "Onomatopoeia"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Cat", "Snake", "Steam", "Onomatopoeia", "Domestic animals, pets", 
       "Wild animals", "Water", "Source-ambiguous sounds", "Animal", 
       "Animal", "Natural sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> <|"Cat" -> 3, "Snake" -> 3, "Steam" -> 3, 
       "Onomatopoeia" -> 2|>, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0011870837968395228, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.113764147862647 - 15.3101794138915*x))^(-1)], 
     "AudioSetID" -> "/m/07rjwbb", "EntityCanonicalName" -> "Hiss::g4k44"|>, 
   "Cat communication" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Cat"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Cat", "Domestic animals, pets", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/0f25s6", "EntityCanonicalName" -> 
      "CatCommunication::7d9xz"|>, "Caterwaul" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> {"Cat"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Cat", "Domestic animals, pets", "Animal"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.00018591042774636943, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.723571190053715 - 24.438216207384894*x))^(-1)], 
     "AudioSetID" -> "/m/07r81j2", "EntityCanonicalName" -> 
      "Caterwaul::5djmg"|>, "Livestock, farm animals, working animals" -> 
    <|"DepthIndex" -> 0.25, "FirstChildren" -> {"Horse", "Donkey, ass", 
       "Cattle, bovinae", "Pig", "Goat", "Sheep", "Fowl"}, 
     "FirstParents" -> {"Animal"}, "FlattenedChildren" -> 
      {"Horse", "Donkey, ass", "Cattle, bovinae", "Pig", "Goat", "Sheep", 
       "Fowl", "Clip-clop", "Neigh, whinny", "Snort (horse)", "Nicker", 
       "Moo", "Cowbell", "Yak", "Oink", "Bleat", "Bleat", "Chicken, rooster", 
       "Turkey", "Duck", "Goose", "Cluck", "Crowing, cock-a-doodle-doo", 
       "Gobble", "Quack", "Honk"}, "FlattenedParents" -> {"Animal"}, 
     "BottomDepth" -> 3, "TopDepth" -> 1, "Restrictions" -> {}, 
     "ClassPrior" -> 0.002109016655745699, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.0083669588202175 - 11.561135933576525*x))^(-1)], 
     "AudioSetID" -> "/m/0ch8v", "EntityCanonicalName" -> 
      "LivestockFarmAnimalsWorkingAnimals::bm569"|>, 
   "Horse" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Clip-clop", "Neigh, whinny", "Snort (horse)", 
       "Nicker"}, "FirstParents" -> 
      {"Livestock, farm animals, working animals"}, 
     "FlattenedChildren" -> {"Clip-clop", "Neigh, whinny", "Snort (horse)", 
       "Nicker"}, "FlattenedParents" -> 
      {"Livestock, farm animals, working animals", "Animal"}, 
     "BottomDepth" -> 1, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0015629682682392861, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.936083032878357 - 10.345956147676782*x))^(-1)], 
     "AudioSetID" -> "/m/03k3r", "EntityCanonicalName" -> "Horse::h6kmk"|>, 
   "Clip-clop" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Horse", "Clicking"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Horse", "Clicking", 
       "Livestock, farm animals, working animals", "Onomatopoeia", "Animal", 
       "Source-ambiguous sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> <|"Horse" -> 3, "Clicking" -> 3|>, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0015903976756117013, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.0152249767713135 - 10.383776866508489*x))^(-1)], 
     "AudioSetID" -> "/m/07rv9rh", "EntityCanonicalName" -> 
      "Clip-clop::2n6md"|>, "Neigh, whinny" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Horse"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Horse", "Livestock, farm animals, working animals", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0002301022507352605, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.636475905675972 - 97.13936485188947*x))^(-1)], 
     "AudioSetID" -> "/m/07q5rw0", "EntityCanonicalName" -> 
      "NeighWhinny::fj9wn"|>, "Snort (horse)" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Horse"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Horse", "Livestock, farm animals, working animals", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/t/dd00139", "EntityCanonicalName" -> 
      "SnortHorse::865k2"|>, "Nicker" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Horse"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Horse", "Livestock, farm animals, working animals", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/t/dd00140", "EntityCanonicalName" -> 
      "Nicker::h8s7q"|>, "Donkey, ass" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> 
      {"Livestock, farm animals, working animals"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Livestock, farm animals, working animals", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/0ffhf", "EntityCanonicalName" -> 
      "DonkeyAss::965xz"|>, "Cattle, bovinae" -> 
    <|"DepthIndex" -> 0.6666666666666666, "FirstChildren" -> 
      {"Moo", "Cowbell", "Yak"}, "FirstParents" -> 
      {"Livestock, farm animals, working animals"}, 
     "FlattenedChildren" -> {"Moo", "Cowbell", "Yak"}, 
     "FlattenedParents" -> {"Livestock, farm animals, working animals", 
       "Animal"}, "BottomDepth" -> 1, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0003454073520970798, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.571125174672126 - 12.311522802160017*x))^(-1)], 
     "AudioSetID" -> "/m/01xq0k1", "EntityCanonicalName" -> 
      "CattleBovinae::32vq9"|>, "Moo" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Cattle, bovinae"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Cattle, bovinae", "Livestock, farm animals, working animals", 
       "Animal"}, "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0002824213055382005, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.807868126221168 - 12.682910367942723*x))^(-1)], 
     "AudioSetID" -> "/m/07rpkh9", "EntityCanonicalName" -> "Moo::y369h"|>, 
   "Cowbell" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Cattle, bovinae", "Percussion", "Bell"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Cattle, bovinae", "Percussion", "Bell", 
       "Livestock, farm animals, working animals", "Musical instrument", 
       "Musical instrument", "Sounds of things", "Animal", "Music", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> <|"Cattle, bovinae" -> 3, 
       "Percussion" -> 3, "Bell" -> 4|>, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00008838364597778218, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.487018124723402 - 18.22857127770064*x))^(-1)], 
     "AudioSetID" -> "/m/0239kh", "EntityCanonicalName" -> 
      "Cowbell::5n775"|>, "Yak" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Cattle, bovinae"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Cattle, bovinae", "Livestock, farm animals, working animals", 
       "Animal"}, "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/01hhp3", "EntityCanonicalName" -> "Yak::344z7"|>, 
   "Pig" -> <|"DepthIndex" -> 0.6666666666666666, "FirstChildren" -> 
      {"Oink"}, "FirstParents" -> 
      {"Livestock, farm animals, working animals"}, 
     "FlattenedChildren" -> {"Oink"}, "FlattenedParents" -> 
      {"Livestock, farm animals, working animals", "Animal"}, 
     "BottomDepth" -> 1, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0004195683424002763, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.27223169838193 - 17.132148974644327*x))^(-1)], 
     "AudioSetID" -> "/m/068zj", "EntityCanonicalName" -> "Pig::5z729"|>, 
   "Oink" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Pig"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Pig", 
       "Livestock, farm animals, working animals", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0007847858220441004, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.4565349688218685 - 14.282342668670482*x))^(-1)], 
     "AudioSetID" -> "/t/dd00018", "EntityCanonicalName" -> "Oink::v5c5y"|>, 
   "Goat" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Bleat"}, "FirstParents" -> 
      {"Livestock, farm animals, working animals"}, 
     "FlattenedChildren" -> {"Bleat"}, "FlattenedParents" -> 
      {"Livestock, farm animals, working animals", "Animal"}, 
     "BottomDepth" -> 1, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0009452986503715668, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.495345036224735 - 11.3723152906659*x))^(-1)], 
     "AudioSetID" -> "/m/03fwl", "EntityCanonicalName" -> "Goat::62f86"|>, 
   "Bleat" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Goat", "Sheep"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Goat", "Sheep", 
       "Livestock, farm animals, working animals", 
       "Livestock, farm animals, working animals", "Animal", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> <|"Goat" -> 3, "Sheep" -> 3|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0009646008259299331, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.221817208163655 - 9.853745892927796*x))^(-1)], 
     "AudioSetID" -> "/m/07q0h5t", "EntityCanonicalName" -> "Bleat::4886t"|>, 
   "Sheep" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Bleat"}, "FirstParents" -> 
      {"Livestock, farm animals, working animals"}, 
     "FlattenedChildren" -> {"Bleat"}, "FlattenedParents" -> 
      {"Livestock, farm animals, working animals", "Animal"}, 
     "BottomDepth" -> 1, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0012536255073170484, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.191238476398141 - 9.650933804634747*x))^(-1)], 
     "AudioSetID" -> "/m/07bgp", "EntityCanonicalName" -> "Sheep::y2w94"|>, 
   "Fowl" -> <|"DepthIndex" -> 0.5, "FirstChildren" -> 
      {"Chicken, rooster", "Turkey", "Duck", "Goose"}, 
     "FirstParents" -> {"Livestock, farm animals, working animals"}, 
     "FlattenedChildren" -> {"Chicken, rooster", "Turkey", "Duck", "Goose", 
       "Cluck", "Crowing, cock-a-doodle-doo", "Gobble", "Quack", "Honk"}, 
     "FlattenedParents" -> {"Livestock, farm animals, working animals", 
       "Animal"}, "BottomDepth" -> 2, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0028450390869055055, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.087778988510384 - 11.023415921867109*x))^(-1)], 
     "AudioSetID" -> "/m/025rv6n", "EntityCanonicalName" -> "Fowl::r5x2g"|>, 
   "Chicken, rooster" -> <|"DepthIndex" -> 0.75, "FirstChildren" -> 
      {"Cluck", "Crowing, cock-a-doodle-doo"}, "FirstParents" -> {"Fowl"}, 
     "FlattenedChildren" -> {"Cluck", "Crowing, cock-a-doodle-doo"}, 
     "FlattenedParents" -> {"Fowl", 
       "Livestock, farm animals, working animals", "Animal"}, 
     "BottomDepth" -> 1, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0029572964763370565, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.210790510114809 - 9.965539530157423*x))^(-1)], 
     "AudioSetID" -> "/m/09b5t", "EntityCanonicalName" -> 
      "ChickenRooster::4623d"|>, "Cluck" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Chicken, rooster"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Chicken, rooster", "Fowl", 
       "Livestock, farm animals, working animals", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.001138320405955229, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.602015393897365 - 12.57293473066654*x))^(-1)], 
     "AudioSetID" -> "/m/07st89h", "EntityCanonicalName" -> "Cluck::4979y"|>, 
   "Crowing, cock-a-doodle-doo" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Chicken, rooster"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Chicken, rooster", "Fowl", 
       "Livestock, farm animals, working animals", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0009417429864529205, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.04087437649034 - 13.299238684562855*x))^(-1)], 
     "AudioSetID" -> "/m/07qn5dc", "EntityCanonicalName" -> 
      "CrowingCock-a-doodle-doo::8863f"|>, 
   "Turkey" -> <|"DepthIndex" -> 0.75, "FirstChildren" -> {"Gobble"}, 
     "FirstParents" -> {"Fowl"}, "FlattenedChildren" -> {"Gobble"}, 
     "FlattenedParents" -> {"Fowl", 
       "Livestock, farm animals, working animals", "Animal"}, 
     "BottomDepth" -> 1, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0005104917483199488, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.385167566590964 - 12.968565857012125*x))^(-1)], 
     "AudioSetID" -> "/m/01rd7k", "EntityCanonicalName" -> "Turkey::6q974"|>, 
   "Gobble" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Turkey"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Turkey", "Fowl", 
       "Livestock, farm animals, working animals", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0003946786949697515, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.673364551326152 - 12.818067002497854*x))^(-1)], 
     "AudioSetID" -> "/m/07svc2k", "EntityCanonicalName" -> 
      "Gobble::5c3zz"|>, "Duck" -> <|"DepthIndex" -> 0.75, 
     "FirstChildren" -> {"Quack"}, "FirstParents" -> {"Fowl"}, 
     "FlattenedChildren" -> {"Quack"}, "FlattenedParents" -> 
      {"Fowl", "Livestock, farm animals, working animals", "Animal"}, 
     "BottomDepth" -> 1, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0012109575402932916, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.394177669541678 - 10.499005489765848*x))^(-1)], 
     "AudioSetID" -> "/m/09ddx", "EntityCanonicalName" -> "Duck::4ccjh"|>, 
   "Quack" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Duck"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Duck", "Fowl", 
       "Livestock, farm animals, working animals", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0011566066775368393, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.253033478360094 - 8.941448257810553*x))^(-1)], 
     "AudioSetID" -> "/m/07qdb04", "EntityCanonicalName" -> "Quack::5jn24"|>, 
   "Goose" -> <|"DepthIndex" -> 0.75, "FirstChildren" -> {"Honk"}, 
     "FirstParents" -> {"Fowl"}, "FlattenedChildren" -> {"Honk"}, 
     "FlattenedParents" -> {"Fowl", 
       "Livestock, farm animals, working animals", "Animal"}, 
     "BottomDepth" -> 1, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.000825421981114345, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.118881343886914 - 10.667414584793345*x))^(-1)], 
     "AudioSetID" -> "/m/0dbvp", "EntityCanonicalName" -> "Goose::sf572"|>, 
   "Honk" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Goose"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Goose", "Fowl", 
       "Livestock, farm animals, working animals", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0008538672924635163, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.035365015509422 - 10.069072468113593*x))^(-1)], 
     "AudioSetID" -> "/m/07qwf61", "EntityCanonicalName" -> "Honk::cwy4p"|>, 
   "Wild animals" -> <|"DepthIndex" -> 0.25, "FirstChildren" -> 
      {"Roaring cats (lions, tigers)", "Bird", "Canidae, dogs, wolves", 
       "Rodents, rats, mice", "Insect", "Frog", "Snake", 
       "Whale vocalization"}, "FirstParents" -> {"Animal"}, 
     "FlattenedChildren" -> {"Roaring cats (lions, tigers)", "Bird", 
       "Canidae, dogs, wolves", "Rodents, rats, mice", "Insect", "Frog", 
       "Snake", "Whale vocalization", "Growling", "Roar", 
       "Bird vocalization, bird call, bird song", "Pigeon, dove", "Crow", 
       "Owl", "Gull, seagull", "Bird flight, flapping wings", "Howl", 
       "Growling", "Mouse", "Chipmunk", "Patter", "Cricket", "Mosquito", 
       "Fly, housefly", "Bee, wasp, etc.", "Croak", "Hiss", "Rattle", 
       "Chirp, tweet", "Squawk", "Coo", "Caw", "Hoot", "Buzz", "Buzz"}, 
     "FlattenedParents" -> {"Animal"}, "BottomDepth" -> 3, "TopDepth" -> 1, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0004952531886686071, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.996056937886162 - 13.680143720381924*x))^(-1)], 
     "AudioSetID" -> "/m/01280g", "EntityCanonicalName" -> 
      "WildAnimals::xk99z"|>, "Roaring cats (lions, tigers)" -> 
    <|"DepthIndex" -> 0.6666666666666666, "FirstChildren" -> 
      {"Roar", "Growling"}, "FirstParents" -> {"Wild animals"}, 
     "FlattenedChildren" -> {"Growling", "Roar"}, "FlattenedParents" -> 
      {"Wild animals", "Animal"}, "BottomDepth" -> 1, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0003174699927362866, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.72023905270166 - 15.246029885789289*x))^(-1)], 
     "AudioSetID" -> "/m/0cdnk", "EntityCanonicalName" -> 
      "RoaringCatsLionsTigers::7n757"|>, 
   "Roar" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Roaring cats (lions, tigers)"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Roaring cats (lions, tigers)", "Wild animals", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0003941707429813734, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.289191241858946 - 15.014887514037408*x))^(-1)], 
     "AudioSetID" -> "/m/04cvmfc", "EntityCanonicalName" -> "Roar::tk62q"|>, 
   "Bird" -> <|"DepthIndex" -> 0.5, "FirstChildren" -> 
      {"Bird vocalization, bird call, bird song", "Pigeon, dove", "Crow", 
       "Owl", "Gull, seagull", "Bird flight, flapping wings"}, 
     "FirstParents" -> {"Wild animals"}, "FlattenedChildren" -> 
      {"Bird vocalization, bird call, bird song", "Pigeon, dove", "Crow", 
       "Owl", "Gull, seagull", "Bird flight, flapping wings", "Chirp, tweet", 
       "Squawk", "Coo", "Caw", "Hoot"}, "FlattenedParents" -> 
      {"Wild animals", "Animal"}, "BottomDepth" -> 2, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.013072144420909336, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.105495785473518 - 8.311259111457515*x))^(-1)], 
     "AudioSetID" -> "/m/015p6", "EntityCanonicalName" -> "Bird::767bs"|>, 
   "Bird vocalization, bird call, bird song" -> <|"DepthIndex" -> 0.75, 
     "FirstChildren" -> {"Chirp, tweet", "Squawk"}, 
     "FirstParents" -> {"Bird"}, "FlattenedChildren" -> 
      {"Chirp, tweet", "Squawk"}, "FlattenedParents" -> 
      {"Bird", "Wild animals", "Animal"}, "BottomDepth" -> 1, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0029730429879767763, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.043936378895386 - 11.13430299868261*x))^(-1)], 
     "AudioSetID" -> "/m/020bb7", "EntityCanonicalName" -> 
      "BirdVocalizationBirdCallBirdSong::4852z"|>, 
   "Chirp, tweet" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Bird vocalization, bird call, bird song", 
       "Brief tone"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Bird vocalization, bird call, bird song", "Brief tone", "Bird", 
       "Onomatopoeia", "Wild animals", "Source-ambiguous sounds", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 
      <|"Bird vocalization, bird call, bird song" -> 4, "Brief tone" -> 3|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.002194860541781591, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.506632132040581 - 12.76597549922182*x))^(-1)], 
     "AudioSetID" -> "/m/07pggtn", "EntityCanonicalName" -> 
      "ChirpTweet::9f9m7"|>, "Squawk" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> 
      {"Bird vocalization, bird call, bird song"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Bird vocalization, bird call, bird song", 
       "Bird", "Wild animals", "Animal"}, "BottomDepth" -> 0, 
     "TopDepth" -> 4, "Restrictions" -> {}, "ClassPrior" -> 
      0.000019810127546744283, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(10.528910030257842 - 762.9009698083617*x))^(-1)], 
     "AudioSetID" -> "/m/07sx8x_", "EntityCanonicalName" -> 
      "Squawk::2rrgf"|>, "Pigeon, dove" -> <|"DepthIndex" -> 0.75, 
     "FirstChildren" -> {"Coo"}, "FirstParents" -> {"Bird"}, 
     "FlattenedChildren" -> {"Coo"}, "FlattenedParents" -> 
      {"Bird", "Wild animals", "Animal"}, "BottomDepth" -> 1, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.004197715231956276, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.4705761172274485 - 9.271729769553133*x))^(-1)], 
     "AudioSetID" -> "/m/0h0rv", "EntityCanonicalName" -> 
      "PigeonDove::9452q"|>, "Coo" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Pigeon, dove"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Pigeon, dove", "Bird", "Wild animals", "Animal"}, "BottomDepth" -> 0, 
     "TopDepth" -> 4, "Restrictions" -> {}, "ClassPrior" -> 
      0.0013958520640629049, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.561296601179148 - 18.69928059417326*x))^(-1)], 
     "AudioSetID" -> "/m/07r_25d", "EntityCanonicalName" -> "Coo::vt6jx"|>, 
   "Crow" -> <|"DepthIndex" -> 0.75, "FirstChildren" -> {"Caw"}, 
     "FirstParents" -> {"Bird"}, "FlattenedChildren" -> {"Caw"}, 
     "FlattenedParents" -> {"Bird", "Wild animals", "Animal"}, 
     "BottomDepth" -> 1, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0003596300077716654, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.22825017042337 - 15.660918924142079*x))^(-1)], 
     "AudioSetID" -> "/m/04s8yn", "EntityCanonicalName" -> "Crow::kjz2y"|>, 
   "Caw" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Crow"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Crow", "Bird", "Wild animals", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00023721357857255332, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.630586929207285 - 15.437215856447269*x))^(-1)], 
     "AudioSetID" -> "/m/07r5c2p", "EntityCanonicalName" -> "Caw::4p65t"|>, 
   "Owl" -> <|"DepthIndex" -> 0.75, "FirstChildren" -> {"Hoot"}, 
     "FirstParents" -> {"Bird"}, "FlattenedChildren" -> {"Hoot"}, 
     "FlattenedParents" -> {"Bird", "Wild animals", "Animal"}, 
     "BottomDepth" -> 1, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00025651575413091957, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.655630674538088 - 35.641477218863365*x))^(-1)], 
     "AudioSetID" -> "/m/09d5_", "EntityCanonicalName" -> "Owl::h8x5r"|>, 
   "Hoot" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Owl"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Owl", "Bird", "Wild animals", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.000024381695442146808, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(11.270767969306585 - 3221.354307546322*x))^(-1)], 
     "AudioSetID" -> "/m/07r_80w", "EntityCanonicalName" -> "Hoot::735x6"|>, 
   "Gull, seagull" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Bird"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Bird", "Wild animals", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/01dwxx", "EntityCanonicalName" -> 
      "GullSeagull::3tnzk"|>, "Bird flight, flapping wings" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> {"Bird"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Bird", "Wild animals", "Animal"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0001102255814780387, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(9.46800671958467 - 71.32374415813864*x))^(-1)], 
     "AudioSetID" -> "/m/05_wcq", "EntityCanonicalName" -> 
      "BirdFlightFlappingWings::x3ntk"|>, "Canidae, dogs, wolves" -> 
    <|"DepthIndex" -> 0.6666666666666666, "FirstChildren" -> 
      {"Howl", "Growling"}, "FirstParents" -> {"Wild animals"}, 
     "FlattenedChildren" -> {"Howl", "Growling"}, "FlattenedParents" -> 
      {"Wild animals", "Animal"}, "BottomDepth" -> 1, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.000789865341927881, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.849690335168366 - 14.787788421532762*x))^(-1)], 
     "AudioSetID" -> "/m/01z5f", "EntityCanonicalName" -> 
      "CanidaeDogsWolves::bkdx2"|>, "Rodents, rats, mice" -> 
    <|"DepthIndex" -> 0.6666666666666666, "FirstChildren" -> 
      {"Mouse", "Chipmunk", "Patter"}, "FirstParents" -> {"Wild animals"}, 
     "FlattenedChildren" -> {"Mouse", "Chipmunk", "Patter"}, 
     "FlattenedParents" -> {"Wild animals", "Animal"}, "BottomDepth" -> 1, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.00025854756208443176, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.39714990267012 - 32.035984644205286*x))^(-1)], 
     "AudioSetID" -> "/m/06hps", "EntityCanonicalName" -> 
      "RodentsRatsMice::3sk2t"|>, "Mouse" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Rodents, rats, mice"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Rodents, rats, mice", "Wild animals", "Animal"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0001457822206645028, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(9.384134230708382 - 483.51234212467466*x))^(-1)], 
     "AudioSetID" -> "/m/04rmv", "EntityCanonicalName" -> "Mouse::2925b"|>, 
   "Chipmunk" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Rodents, rats, mice"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Rodents, rats, mice", "Wild animals", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/02021", "EntityCanonicalName" -> 
      "Chipmunk::y7z8z"|>, "Patter" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Rodents, rats, mice"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Rodents, rats, mice", "Wild animals", "Animal"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> 0.0002915644413290056, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.32150057670713 - 87.41034020767073*x))^(-1)], 
     "AudioSetID" -> "/m/07r4gkf", "EntityCanonicalName" -> 
      "Patter::6d532"|>, "Insect" -> <|"DepthIndex" -> 0.5, 
     "FirstChildren" -> {"Cricket", "Mosquito", "Fly, housefly", 
       "Bee, wasp, etc."}, "FirstParents" -> {"Wild animals"}, 
     "FlattenedChildren" -> {"Cricket", "Mosquito", "Fly, housefly", 
       "Bee, wasp, etc.", "Buzz", "Buzz"}, "FlattenedParents" -> 
      {"Wild animals", "Animal"}, "BottomDepth" -> 2, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0013943282080977706, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.343920551622634 - 10.928748626444209*x))^(-1)], 
     "AudioSetID" -> "/m/03vt0", "EntityCanonicalName" -> "Insect::68mpj"|>, 
   "Cricket" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Insect"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Insect", "Wild animals", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0003586141037949093, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.323961562417288 - 15.91577671871598*x))^(-1)], 
     "AudioSetID" -> "/m/09xqv", "EntityCanonicalName" -> "Cricket::pnfm8"|>, 
   "Mosquito" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Insect"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Insect", "Wild animals", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00010870172551290452, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.382354202843937 - 15.638574959001117*x))^(-1)], 
     "AudioSetID" -> "/m/09f96", "EntityCanonicalName" -> 
      "Mosquito::x4379"|>, "Fly, housefly" -> <|"DepthIndex" -> 0.75, 
     "FirstChildren" -> {"Buzz"}, "FirstParents" -> {"Insect"}, 
     "FlattenedChildren" -> {"Buzz"}, "FlattenedParents" -> 
      {"Insect", "Wild animals", "Animal"}, "BottomDepth" -> 1, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0006445910732517562, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.097471359701954 - 11.236047672375955*x))^(-1)], 
     "AudioSetID" -> "/m/0h2mp", "EntityCanonicalName" -> 
      "FlyHousefly::5kkbc"|>, "Buzz" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Fly, housefly", 
       "Bee, wasp, etc.", "Brief tone"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Fly, housefly", "Bee, wasp, etc.", "Brief tone", 
       "Insect", "Insect", "Onomatopoeia", "Wild animals", "Wild animals", 
       "Source-ambiguous sounds", "Animal", "Animal"}, "BottomDepth" -> 0, 
     "TopDepth" -> <|"Fly, housefly" -> 4, "Bee, wasp, etc." -> 4, 
       "Brief tone" -> 3|>, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00015898897236233232, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.909824979897198 - 227.0124922383164*x))^(-1)], 
     "AudioSetID" -> "/m/07pjwq1", "EntityCanonicalName" -> "Buzz::8c2x2"|>, 
   "Bee, wasp, etc." -> <|"DepthIndex" -> 0.75, "FirstChildren" -> {"Buzz"}, 
     "FirstParents" -> {"Insect"}, "FlattenedChildren" -> {"Buzz"}, 
     "FlattenedParents" -> {"Insect", "Wild animals", "Animal"}, 
     "BottomDepth" -> 1, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0008665660921729679, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.143556939876058 - 11.43788219533729*x))^(-1)], 
     "AudioSetID" -> "/m/01h3n", "EntityCanonicalName" -> 
      "BeeWaspEtc.::rm568"|>, "Frog" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Croak"}, "FirstParents" -> {"Wild animals"}, 
     "FlattenedChildren" -> {"Croak"}, "FlattenedParents" -> 
      {"Wild animals", "Animal"}, "BottomDepth" -> 1, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.000555191523297218, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.291676906090254 - 9.830691067194788*x))^(-1)], 
     "AudioSetID" -> "/m/09ld4", "EntityCanonicalName" -> "Frog::p2q8d"|>, 
   "Croak" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Frog"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Frog", "Wild animals", "Animal"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00012800390107127075, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.50479907952635 - 13.945031398572912*x))^(-1)], 
     "AudioSetID" -> "/m/07st88b", "EntityCanonicalName" -> "Croak::g9vbc"|>, 
   "Snake" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Hiss", "Rattle"}, "FirstParents" -> 
      {"Wild animals"}, "FlattenedChildren" -> {"Hiss", "Rattle"}, 
     "FlattenedParents" -> {"Wild animals", "Animal"}, "BottomDepth" -> 1, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0005465563394947909, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.8821902522339276 - 26.798125833288832*x))^(-1)], 
     "AudioSetID" -> "/m/078jl", "EntityCanonicalName" -> "Snake::59r7k"|>, 
   "Rattle" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Snake", "Onomatopoeia"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Snake", "Onomatopoeia", "Wild animals", 
       "Source-ambiguous sounds", "Animal"}, "BottomDepth" -> 0, 
     "TopDepth" -> <|"Snake" -> 3, "Onomatopoeia" -> 2|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0004739192051567286, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.854165201666715 - 49.42949559736709*x))^(-1)], 
     "AudioSetID" -> "/m/07qn4z3", "EntityCanonicalName" -> 
      "Rattle::qsc6p"|>, "Whale vocalization" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Wild animals"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Wild animals", "Animal"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00006247809457050119, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.72061855245095 - 204.38688070535795*x))^(-1)], 
     "AudioSetID" -> "/m/032n05", "EntityCanonicalName" -> 
      "WhaleVocalization::2zc33"|>, "Music" -> <|"DepthIndex" -> 0., 
     "FirstChildren" -> {"Musical instrument", "Music genre", 
       "Musical concepts", "Music role", "Music mood"}, "FirstParents" -> {}, 
     "FlattenedChildren" -> {"Musical instrument", "Music genre", 
       "Musical concepts", "Music role", "Music mood", "Choir", 
       "Plucked string instrument", "Keyboard (musical)", "Percussion", 
       "Orchestra", "Brass instrument", "Bowed string instrument", 
       "Wind instrument, woodwind instrument", "Harp", "Bell", "Harmonica", 
       "Accordion", "Bagpipes", "Didgeridoo", "Shofar", "Theremin", 
       "Singing bowl", "Musical ensemble", "Bass (instrument role)", 
       "Scratching (performance technique)", "Pop music", "Hip hop music", 
       "Rock music", "Rhythm and blues", "Soul music", "Reggae", "Country", 
       "Funk", "Folk music", "Middle Eastern music", "Jazz", "Disco", 
       "Classical music", "Electronic music", "Music of Latin America", 
       "Blues", "Music for children", "New-age music", "Vocal music", 
       "Music of Africa", "Christian music", "Music of Asia", "Ska", 
       "Traditional music", "Independent music", "Song", "Melody", 
       "Musical note", "Beat", "Chord", "Harmony", "Bassline", "Loop", 
       "Drone", "Background music", "Theme music", "Jingle (music)", 
       "Soundtrack music", "Lullaby", "Video game music", "Christmas music", 
       "Dance music", "Wedding music", "Birthday music", "Happy music", 
       "Funny music", "Sad music", "Tender music", "Exciting music", 
       "Angry music", "Scary music", "Guitar", "Banjo", "Sitar", "Mandolin", 
       "Zither", "Ukulele", "Piano", "Organ", "Synthesizer", "Harpsichord", 
       "Cowbell", "Drum kit", "Drum", "Cymbal", "Wood block", "Tambourine", 
       "Rattle (instrument)", "Gong", "Tubular bells", "Mallet percussion", 
       "French horn", "Trumpet", "Trombone", "Cornet", "Bugle", 
       "String section", "Violin, fiddle", "Cello", "Double bass", "Flute", 
       "Saxophone", "Clarinet", "Oboe", "Bassoon", "Cowbell", "Church bell", 
       "Jingle bell", "Bicycle bell", "Tuning fork", "Chime", 
       "Change ringing (campanology)", "Grime music", "Trap music", 
       "Beatboxing", "Heavy metal", "Punk rock", "Grunge", 
       "Progressive rock", "Rock and roll", "Psychedelic rock", "Dub", 
       "Swing music", "Bluegrass", "Opera", "House music", "Techno", 
       "Dubstep", "Electro", "Drum and bass", "Electronica", 
       "Electronic dance music", "Ambient music", "Trance music", 
       "Noise music", "UK garage", "Cumbia", "Salsa music", "Soca music", 
       "Kuduro", "Funk carioca", "Flamenco", "Chant", "Beatboxing", 
       "A capella", "Afrobeat", "Kwaito", "Gospel music", "Carnatic music", 
       "Music of Bollywood", "Drum beat", "Electric guitar", "Bass guitar", 
       "Acoustic guitar", "Steel guitar, slide guitar", 
       "Tapping (guitar technique)", "Strum", "Electric piano", 
       "Electronic organ", "Hammond organ", "Sampler", "Mellotron", 
       "Drum machine", "Snare drum", "Bass drum", "Timpani", "Tabla", 
       "Hi-hat", "Crash cymbal", "Maraca", "Marimba, xylophone", 
       "Glockenspiel", "Vibraphone", "Steelpan", "Pizzicato", 
       "Alto saxophone", "Soprano saxophone", "Wind chime", 
       "Oldschool jungle", "Drone music", "Mantra", "Clavinet", 
       "Rhodes piano", "Rimshot", "Drum roll"}, "FlattenedParents" -> {}, 
     "BottomDepth" -> 5, "TopDepth" -> 0, "Restrictions" -> {}, 
     "ClassPrior" -> 0.4883577404263749, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(10.463027524117058 - 13.074684676521038*x))^(-1)], 
     "AudioSetID" -> "/m/04rlf", "EntityCanonicalName" -> "Music::s3g22"|>, 
   "Musical instrument" -> <|"DepthIndex" -> 0.2, "FirstChildren" -> 
      {"Plucked string instrument", "Keyboard (musical)", "Percussion", 
       "Orchestra", "Brass instrument", "Bowed string instrument", 
       "Wind instrument, woodwind instrument", "Harp", "Choir", "Bell", 
       "Harmonica", "Accordion", "Bagpipes", "Didgeridoo", "Shofar", 
       "Theremin", "Singing bowl", "Musical ensemble", 
       "Bass (instrument role)", "Scratching (performance technique)"}, 
     "FirstParents" -> {"Music"}, "FlattenedChildren" -> 
      {"Choir", "Plucked string instrument", "Keyboard (musical)", 
       "Percussion", "Orchestra", "Brass instrument", 
       "Bowed string instrument", "Wind instrument, woodwind instrument", 
       "Harp", "Bell", "Harmonica", "Accordion", "Bagpipes", "Didgeridoo", 
       "Shofar", "Theremin", "Singing bowl", "Musical ensemble", 
       "Bass (instrument role)", "Scratching (performance technique)", 
       "Guitar", "Banjo", "Sitar", "Mandolin", "Zither", "Ukulele", "Piano", 
       "Organ", "Synthesizer", "Harpsichord", "Cowbell", "Drum kit", "Drum", 
       "Cymbal", "Wood block", "Tambourine", "Rattle (instrument)", "Gong", 
       "Tubular bells", "Mallet percussion", "French horn", "Trumpet", 
       "Trombone", "Cornet", "Bugle", "String section", "Violin, fiddle", 
       "Cello", "Double bass", "Flute", "Saxophone", "Clarinet", "Oboe", 
       "Bassoon", "Cowbell", "Church bell", "Jingle bell", "Bicycle bell", 
       "Tuning fork", "Chime", "Change ringing (campanology)", 
       "Electric guitar", "Bass guitar", "Acoustic guitar", 
       "Steel guitar, slide guitar", "Tapping (guitar technique)", "Strum", 
       "Electric piano", "Electronic organ", "Hammond organ", "Sampler", 
       "Mellotron", "Drum machine", "Snare drum", "Bass drum", "Timpani", 
       "Tabla", "Hi-hat", "Crash cymbal", "Maraca", "Marimba, xylophone", 
       "Glockenspiel", "Vibraphone", "Steelpan", "Pizzicato", 
       "Alto saxophone", "Soprano saxophone", "Wind chime", "Clavinet", 
       "Rhodes piano", "Rimshot", "Drum roll"}, "FlattenedParents" -> 
      {"Music"}, "BottomDepth" -> 4, "TopDepth" -> 1, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0578364293007025, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(4.871491301920043 - 6.620472281576508*x))^(-1)], 
     "AudioSetID" -> "/m/04szw", "EntityCanonicalName" -> 
      "MusicalInstrument::f8frf"|>, "Plucked string instrument" -> 
    <|"DepthIndex" -> 0.5, "FirstChildren" -> {"Guitar", "Banjo", "Sitar", 
       "Mandolin", "Zither", "Ukulele"}, "FirstParents" -> 
      {"Musical instrument"}, "FlattenedChildren" -> 
      {"Guitar", "Banjo", "Sitar", "Mandolin", "Zither", "Ukulele", 
       "Electric guitar", "Bass guitar", "Acoustic guitar", 
       "Steel guitar, slide guitar", "Tapping (guitar technique)", "Strum"}, 
     "FlattenedParents" -> {"Musical instrument", "Music"}, 
     "BottomDepth" -> 2, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.02181704585282599, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.025753443904638 - 7.768748873954279*x))^(-1)], 
     "AudioSetID" -> "/m/0fx80y", "EntityCanonicalName" -> 
      "PluckedStringInstrument::hr6qj"|>, 
   "Guitar" -> <|"DepthIndex" -> 0.75, "FirstChildren" -> 
      {"Electric guitar", "Bass guitar", "Acoustic guitar", 
       "Steel guitar, slide guitar", "Tapping (guitar technique)", "Strum"}, 
     "FirstParents" -> {"Plucked string instrument"}, 
     "FlattenedChildren" -> {"Electric guitar", "Bass guitar", 
       "Acoustic guitar", "Steel guitar, slide guitar", 
       "Tapping (guitar technique)", "Strum"}, "FlattenedParents" -> 
      {"Plucked string instrument", "Musical instrument", "Music"}, 
     "BottomDepth" -> 1, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.025262484189994362, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(5.971454134889372 - 7.364809376270449*x))^(-1)], 
     "AudioSetID" -> "/m/0342h", "EntityCanonicalName" -> "Guitar::pt485"|>, 
   "Electric guitar" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Guitar"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Guitar", "Plucked string instrument", 
       "Musical instrument", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 4, 
     "Restrictions" -> {}, "ClassPrior" -> 0.005862781849859551, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.544716911832502 - 10.829316688636037*x))^(-1)], 
     "AudioSetID" -> "/m/02sgy", "EntityCanonicalName" -> 
      "ElectricGuitar::5984w"|>, "Bass guitar" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Guitar"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Guitar", "Plucked string instrument", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.003125428584490194, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.459816617010281 - 9.505009504446353*x))^(-1)], 
     "AudioSetID" -> "/m/018vs", "EntityCanonicalName" -> 
      "BassGuitar::2427j"|>, "Acoustic guitar" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Guitar"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Guitar", "Plucked string instrument", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.007128090252909295, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.633954784006521 - 9.159016505831271*x))^(-1)], 
     "AudioSetID" -> "/m/042v_gx", "EntityCanonicalName" -> 
      "AcousticGuitar::p65gm"|>, "Steel guitar, slide guitar" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Guitar"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Guitar", "Plucked string instrument", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0013435330092599648, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.353618124375637 - 12.034252904403159*x))^(-1)], 
     "AudioSetID" -> "/m/06w87", "EntityCanonicalName" -> 
      "SteelGuitarSlideGuitar::fwpd5"|>, "Tapping (guitar technique)" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Guitar"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Guitar", "Plucked string instrument", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0008360889728702844, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.651681982385482 - 24.34250730799901*x))^(-1)], 
     "AudioSetID" -> "/m/01glhc", "EntityCanonicalName" -> 
      "TappingGuitarTechnique::fxm3m"|>, 
   "Strum" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Guitar"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Guitar", "Plucked string instrument", 
       "Musical instrument", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 4, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00685989160304568, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.322116907705018 - 10.18148046909507*x))^(-1)], 
     "AudioSetID" -> "/m/07s0s5r", "EntityCanonicalName" -> "Strum::jxt32"|>, 
   "Banjo" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Plucked string instrument"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Plucked string instrument", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0011693054772462907, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.03231503003499 - 11.514721730429704*x))^(-1)], 
     "AudioSetID" -> "/m/018j2", "EntityCanonicalName" -> "Banjo::ry37k"|>, 
   "Sitar" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Plucked string instrument"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Plucked string instrument", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0007146884476479283, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.655583085156549 - 10.86757554945524*x))^(-1)], 
     "AudioSetID" -> "/m/0jtg0", "EntityCanonicalName" -> "Sitar::gnt94"|>, 
   "Mandolin" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Plucked string instrument"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Plucked string instrument", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.001128161366187668, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.040669113359678 - 10.202333927128631*x))^(-1)], 
     "AudioSetID" -> "/m/04rzd", "EntityCanonicalName" -> 
      "Mandolin::3hsy3"|>, "Zither" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Plucked string instrument"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Plucked string instrument", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.000538429107680742, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.08371326232178 - 27.102431243691928*x))^(-1)], 
     "AudioSetID" -> "/m/01bns_", "EntityCanonicalName" -> "Zither::546p6"|>, 
   "Ukulele" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Plucked string instrument"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Plucked string instrument", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.002570745013181354, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.774300099851085 - 10.682558503292103*x))^(-1)], 
     "AudioSetID" -> "/m/07xzm", "EntityCanonicalName" -> "Ukulele::4m234"|>, 
   "Keyboard (musical)" -> <|"DepthIndex" -> 0.4, "FirstChildren" -> 
      {"Piano", "Organ", "Synthesizer", "Harpsichord"}, 
     "FirstParents" -> {"Musical instrument"}, "FlattenedChildren" -> 
      {"Piano", "Organ", "Synthesizer", "Harpsichord", "Electric piano", 
       "Electronic organ", "Hammond organ", "Sampler", "Mellotron", 
       "Clavinet", "Rhodes piano"}, "FlattenedParents" -> 
      {"Musical instrument", "Music"}, "BottomDepth" -> 3, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.005040915532663852, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.264685954134855 - 10.986663589345895*x))^(-1)], 
     "AudioSetID" -> "/m/05148p4", "EntityCanonicalName" -> 
      "KeyboardMusical::53k2f"|>, "Piano" -> 
    <|"DepthIndex" -> 0.6000000000000001, "FirstChildren" -> 
      {"Electric piano"}, "FirstParents" -> {"Keyboard (musical)"}, 
     "FlattenedChildren" -> {"Electric piano", "Clavinet", "Rhodes piano"}, 
     "FlattenedParents" -> {"Keyboard (musical)", "Musical instrument", 
       "Music"}, "BottomDepth" -> 2, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.005624552367310242, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.314800141260504 - 10.64907271777047*x))^(-1)], 
     "AudioSetID" -> "/m/05r5c", "EntityCanonicalName" -> "Piano::h2636"|>, 
   "Electric piano" -> <|"DepthIndex" -> 0.8, "FirstChildren" -> 
      {"Clavinet", "Rhodes piano"}, "FirstParents" -> {"Piano"}, 
     "FlattenedChildren" -> {"Clavinet", "Rhodes piano"}, 
     "FlattenedParents" -> {"Piano", "Keyboard (musical)", 
       "Musical instrument", "Music"}, "BottomDepth" -> 1, "TopDepth" -> 4, 
     "Restrictions" -> {}, "ClassPrior" -> 0.002254798876410202, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.749585539735943 - 13.182749598679006*x))^(-1)], 
     "AudioSetID" -> "/m/01s0ps", "EntityCanonicalName" -> 
      "ElectricPiano::bq943"|>, "Clavinet" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Electric piano"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Electric piano", "Piano", "Keyboard (musical)", "Musical instrument", 
       "Music"}, "BottomDepth" -> 0, "TopDepth" -> 5, 
     "Restrictions" -> {"blacklist"}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/025cbm", "EntityCanonicalName" -> 
      "Clavinet::zsbn6"|>, "Rhodes piano" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Electric piano"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Electric piano", "Piano", "Keyboard (musical)", "Musical instrument", 
       "Music"}, "BottomDepth" -> 0, "TopDepth" -> 5, 
     "Restrictions" -> {"blacklist"}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], "AudioSetID" -> "/m/0bxl5", 
     "EntityCanonicalName" -> "RhodesPiano::55x32"|>, 
   "Organ" -> <|"DepthIndex" -> 0.75, "FirstChildren" -> 
      {"Electronic organ", "Hammond organ"}, "FirstParents" -> 
      {"Keyboard (musical)"}, "FlattenedChildren" -> 
      {"Electronic organ", "Hammond organ"}, "FlattenedParents" -> 
      {"Keyboard (musical)", "Musical instrument", "Music"}, 
     "BottomDepth" -> 1, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0013506443370972575, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.38890956583516 - 11.754299745783971*x))^(-1)], 
     "AudioSetID" -> "/m/013y1f", "EntityCanonicalName" -> "Organ::jfw6p"|>, 
   "Electronic organ" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Organ"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Organ", "Keyboard (musical)", 
       "Musical instrument", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 4, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0005953197303790845, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.949594054228548 - 16.88670401454434*x))^(-1)], 
     "AudioSetID" -> "/m/03xq_f", "EntityCanonicalName" -> 
      "ElectronicOrgan::96j77"|>, "Hammond organ" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> {"Organ"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Organ", "Keyboard (musical)", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0006303684175771706, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.118140939817467 - 16.113283752942944*x))^(-1)], 
     "AudioSetID" -> "/m/03gvt", "EntityCanonicalName" -> 
      "HammondOrgan::8rjzs"|>, "Synthesizer" -> <|"DepthIndex" -> 0.75, 
     "FirstChildren" -> {"Sampler", "Mellotron"}, 
     "FirstParents" -> {"Keyboard (musical)"}, "FlattenedChildren" -> 
      {"Sampler", "Mellotron"}, "FlattenedParents" -> 
      {"Keyboard (musical)", "Musical instrument", "Music"}, 
     "BottomDepth" -> 1, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.002449852439947376, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.682510730303144 - 10.579648273544821*x))^(-1)], 
     "AudioSetID" -> "/m/0l14qv", "EntityCanonicalName" -> 
      "Synthesizer::8tgf3"|>, "Sampler" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Synthesizer"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Synthesizer", "Keyboard (musical)", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.002638302627635636, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.27828719409848 - 14.924365756772504*x))^(-1)], 
     "AudioSetID" -> "/m/01v1d8", "EntityCanonicalName" -> 
      "Sampler::b2sf7"|>, "Mellotron" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Synthesizer"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Synthesizer", "Keyboard (musical)", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/0gkd1", "EntityCanonicalName" -> 
      "Mellotron::j8nmt"|>, "Harpsichord" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Keyboard (musical)"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Keyboard (musical)", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0009458066023599449, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.861533473830491 - 11.535350818634495*x))^(-1)], 
     "AudioSetID" -> "/m/03q5t", "EntityCanonicalName" -> 
      "Harpsichord::x8yh9"|>, "Percussion" -> <|"DepthIndex" -> 0.4, 
     "FirstChildren" -> {"Drum kit", "Drum", "Cymbal", "Cowbell", 
       "Wood block", "Tambourine", "Rattle (instrument)", "Gong", 
       "Tubular bells", "Mallet percussion"}, "FirstParents" -> 
      {"Musical instrument"}, "FlattenedChildren" -> 
      {"Cowbell", "Drum kit", "Drum", "Cymbal", "Wood block", "Tambourine", 
       "Rattle (instrument)", "Gong", "Tubular bells", "Mallet percussion", 
       "Drum machine", "Snare drum", "Bass drum", "Timpani", "Tabla", 
       "Hi-hat", "Crash cymbal", "Maraca", "Marimba, xylophone", 
       "Glockenspiel", "Vibraphone", "Steelpan", "Rimshot", "Drum roll"}, 
     "FlattenedParents" -> {"Musical instrument", "Music"}, 
     "BottomDepth" -> 3, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00818412243674728, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.247967147459929 - 10.003538841500838*x))^(-1)], 
     "AudioSetID" -> "/m/0l14md", "EntityCanonicalName" -> 
      "Percussion::h7v95"|>, "Drum kit" -> <|"DepthIndex" -> 0.75, 
     "FirstChildren" -> {"Drum machine"}, "FirstParents" -> {"Percussion"}, 
     "FlattenedChildren" -> {"Drum machine"}, "FlattenedParents" -> 
      {"Percussion", "Musical instrument", "Music"}, "BottomDepth" -> 1, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.007447084101610716, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.1220662648005 - 10.402712133604462*x))^(-1)], 
     "AudioSetID" -> "/m/02hnl", "EntityCanonicalName" -> "DrumKit::g3283"|>, 
   "Drum machine" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Drum kit"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Drum kit", "Percussion", "Musical instrument", 
       "Music"}, "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0015792227318673839, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.129449130981113 - 15.827308650863383*x))^(-1)], 
     "AudioSetID" -> "/m/0cfdd", "EntityCanonicalName" -> 
      "DrumMachine::sq49v"|>, "Drum" -> <|"DepthIndex" -> 0.6000000000000001, 
     "FirstChildren" -> {"Snare drum", "Bass drum", "Timpani", "Tabla"}, 
     "FirstParents" -> {"Percussion"}, "FlattenedChildren" -> 
      {"Snare drum", "Bass drum", "Timpani", "Tabla", "Rimshot", 
       "Drum roll"}, "FlattenedParents" -> {"Percussion", 
       "Musical instrument", "Music"}, "BottomDepth" -> 2, "TopDepth" -> 3, 
     "Restrictions" -> {}, "ClassPrior" -> 0.009856300382487847, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(5.9253908144806235 - 9.588988795916457*x))^(-1)], 
     "AudioSetID" -> "/m/026t6", "EntityCanonicalName" -> "Drum::mx483"|>, 
   "Snare drum" -> <|"DepthIndex" -> 0.8, "FirstChildren" -> 
      {"Rimshot", "Drum roll"}, "FirstParents" -> {"Drum"}, 
     "FlattenedChildren" -> {"Rimshot", "Drum roll"}, 
     "FlattenedParents" -> {"Drum", "Percussion", "Musical instrument", 
       "Music"}, "BottomDepth" -> 1, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0033072753963295388, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.123746525260048 - 12.181084550611521*x))^(-1)], 
     "AudioSetID" -> "/m/06rvn", "EntityCanonicalName" -> 
      "SnareDrum::9xmf8"|>, "Rimshot" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Snare drum"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Snare drum", "Drum", "Percussion", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 5, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0021669231824207975, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.157096586569251 - 19.718958996938543*x))^(-1)], 
     "AudioSetID" -> "/m/03t3fj", "EntityCanonicalName" -> 
      "Rimshot::38w62"|>, "Drum roll" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Snare drum"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Snare drum", "Drum", "Percussion", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 5, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0014100747197374904, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.520021460121225 - 17.976163436106376*x))^(-1)], 
     "AudioSetID" -> "/m/02k_mr", "EntityCanonicalName" -> 
      "DrumRoll::9kbym"|>, "Bass drum" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Drum"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Drum", "Percussion", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.004525344264460124, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.659049703522318 - 12.603247701059296*x))^(-1)], 
     "AudioSetID" -> "/m/0bm02", "EntityCanonicalName" -> 
      "BassDrum::3nd28"|>, "Timpani" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Drum"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Drum", "Percussion", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.000616145761902585, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(9.164138416272296 - 11.756478839402337*x))^(-1)], 
     "AudioSetID" -> "/m/011k_j", "EntityCanonicalName" -> 
      "Timpani::4y48z"|>, "Tabla" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Drum"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Drum", "Percussion", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0007837699180673443, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.429639767665712 - 10.415137066041513*x))^(-1)], 
     "AudioSetID" -> "/m/01p970", "EntityCanonicalName" -> "Tabla::mz3rm"|>, 
   "Cymbal" -> <|"DepthIndex" -> 0.75, "FirstChildren" -> 
      {"Hi-hat", "Crash cymbal"}, "FirstParents" -> {"Percussion"}, 
     "FlattenedChildren" -> {"Hi-hat", "Crash cymbal"}, 
     "FlattenedParents" -> {"Percussion", "Musical instrument", "Music"}, 
     "BottomDepth" -> 1, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0022542909244218238, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.29591297040296 - 12.32102276697808*x))^(-1)], 
     "AudioSetID" -> "/m/01qbl", "EntityCanonicalName" -> "Cymbal::s6x67"|>, 
   "Hi-hat" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Cymbal"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Cymbal", "Percussion", "Musical instrument", 
       "Music"}, "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0018707871731963896, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.421151820489162 - 13.652343481701957*x))^(-1)], 
     "AudioSetID" -> "/m/03qtq", "EntityCanonicalName" -> "Hi-hat::7q4cw"|>, 
   "Crash cymbal" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Cymbal"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Cymbal", "Percussion", "Musical instrument", 
       "Music"}, "BottomDepth" -> 0, "TopDepth" -> 4, 
     "Restrictions" -> {"blacklist"}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], "AudioSetID" -> "/m/0bm0k", 
     "EntityCanonicalName" -> "CrashCymbal::3q36k"|>, 
   "Wood block" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Percussion"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Percussion", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0010087926489188242, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.641794857403882 - 15.760146191835977*x))^(-1)], 
     "AudioSetID" -> "/m/01sm1g", "EntityCanonicalName" -> 
      "WoodBlock::d3cr7"|>, "Tambourine" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Percussion"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Percussion", "Musical instrument", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0003098507129106157, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.184683008915089 - 14.33131636054302*x))^(-1)], 
     "AudioSetID" -> "/m/07brj", "EntityCanonicalName" -> 
      "Tambourine::578k7"|>, "Rattle (instrument)" -> 
    <|"DepthIndex" -> 0.75, "FirstChildren" -> {"Maraca"}, 
     "FirstParents" -> {"Percussion"}, "FlattenedChildren" -> {"Maraca"}, 
     "FlattenedParents" -> {"Percussion", "Musical instrument", "Music"}, 
     "BottomDepth" -> 1, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00013765498885045385, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.450563008708377 - 20.31223533778686*x))^(-1)], 
     "AudioSetID" -> "/m/05r5wn", "EntityCanonicalName" -> 
      "RattleInstrument::rv8p9"|>, "Maraca" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Rattle (instrument)"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Rattle (instrument)", "Percussion", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00023670562658417526, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.483163182411618 - 24.67681356700479*x))^(-1)], 
     "AudioSetID" -> "/m/0xzly", "EntityCanonicalName" -> "Maraca::krn59"|>, 
   "Gong" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Percussion"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Percussion", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.000182354763827723, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(9.387923083993993 - 18.457850797256476*x))^(-1)], 
     "AudioSetID" -> "/m/0mbct", "EntityCanonicalName" -> "Gong::288k9"|>, 
   "Tubular bells" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Percussion"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Percussion", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00019556151552555252, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.842285733866445 - 59.590526689768204*x))^(-1)], 
     "AudioSetID" -> "/m/016622", "EntityCanonicalName" -> 
      "TubularBells::9z564"|>, "Mallet percussion" -> 
    <|"DepthIndex" -> 0.75, "FirstChildren" -> {"Marimba, xylophone", 
       "Glockenspiel", "Vibraphone", "Steelpan"}, 
     "FirstParents" -> {"Percussion"}, "FlattenedChildren" -> 
      {"Marimba, xylophone", "Glockenspiel", "Vibraphone", "Steelpan"}, 
     "FlattenedParents" -> {"Percussion", "Musical instrument", "Music"}, 
     "BottomDepth" -> 1, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0015238559651341754, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.407942993249844 - 17.903772538105702*x))^(-1)], 
     "AudioSetID" -> "/m/0j45pbj", "EntityCanonicalName" -> 
      "MalletPercussion::368f9"|>, "Marimba, xylophone" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Mallet percussion"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Mallet percussion", "Percussion", 
       "Musical instrument", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 4, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0026347469637169893, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.3976293034058935 - 12.607085639005696*x))^(-1)], 
     "AudioSetID" -> "/m/0dwsp", "EntityCanonicalName" -> 
      "MarimbaXylophone::p3bc8"|>, "Glockenspiel" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Mallet percussion"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Mallet percussion", "Percussion", 
       "Musical instrument", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 4, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0019479958754298544, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.309841595344066 - 17.311781055077237*x))^(-1)], 
     "AudioSetID" -> "/m/0dwtp", "EntityCanonicalName" -> 
      "Glockenspiel::cxp8y"|>, "Vibraphone" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Mallet percussion"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Mallet percussion", "Percussion", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0006923385601592938, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.413054885549025 - 15.243286410792354*x))^(-1)], 
     "AudioSetID" -> "/m/0dwt5", "EntityCanonicalName" -> 
      "Vibraphone::7xc5s"|>, "Steelpan" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Mallet percussion"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Mallet percussion", "Percussion", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00043379099807486194, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.212770089711812 - 12.826994814950606*x))^(-1)], 
     "AudioSetID" -> "/m/0l156b", "EntityCanonicalName" -> 
      "Steelpan::hz5km"|>, "Orchestra" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Musical instrument"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Musical instrument", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00413625304136253, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.58336567546325 - 10.852607954225732*x))^(-1)], 
     "AudioSetID" -> "/m/05pd6", "EntityCanonicalName" -> 
      "Orchestra::dnhv5"|>, "Brass instrument" -> 
    <|"DepthIndex" -> 0.6666666666666666, "FirstChildren" -> 
      {"French horn", "Trumpet", "Trombone", "Cornet", "Bugle"}, 
     "FirstParents" -> {"Musical instrument"}, "FlattenedChildren" -> 
      {"French horn", "Trumpet", "Trombone", "Cornet", "Bugle"}, 
     "FlattenedParents" -> {"Musical instrument", "Music"}, 
     "BottomDepth" -> 1, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.003516551615541299, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.402907644534747 - 10.377903810126666*x))^(-1)], 
     "AudioSetID" -> "/m/01kcd", "EntityCanonicalName" -> 
      "BrassInstrument::t5yf4"|>, "French horn" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Brass instrument"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Brass instrument", "Musical instrument", 
       "Music"}, "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0005424927235877665, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.675263611551253 - 16.594731075458217*x))^(-1)], 
     "AudioSetID" -> "/m/0319l", "EntityCanonicalName" -> 
      "FrenchHorn::6638r"|>, "Trumpet" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Brass instrument"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Brass instrument", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0018469134297426208, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.722815259930712 - 11.578989164676642*x))^(-1)], 
     "AudioSetID" -> "/m/07gql", "EntityCanonicalName" -> "Trumpet::527zq"|>, 
   "Trombone" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Brass instrument"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Brass instrument", "Musical instrument", 
       "Music"}, "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0013267705936434889, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.067672738853307 - 12.695798082096061*x))^(-1)], 
     "AudioSetID" -> "/m/07c6l", "EntityCanonicalName" -> 
      "Trombone::zf2pj"|>, "Cornet" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Brass instrument"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Brass instrument", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/020w2", "EntityCanonicalName" -> "Cornet::f29dt"|>, 
   "Bugle" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Brass instrument"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Brass instrument", "Musical instrument", 
       "Music"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {"blacklist"}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], "AudioSetID" -> "/m/0y64j", 
     "EntityCanonicalName" -> "Bugle::3gm76"|>, "Bowed string instrument" -> 
    <|"DepthIndex" -> 0.5, "FirstChildren" -> {"String section", 
       "Violin, fiddle", "Cello", "Double bass"}, 
     "FirstParents" -> {"Musical instrument"}, "FlattenedChildren" -> 
      {"String section", "Violin, fiddle", "Cello", "Double bass", 
       "Pizzicato"}, "FlattenedParents" -> {"Musical instrument", "Music"}, 
     "BottomDepth" -> 2, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.005016025885233328, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.344670745142417 - 12.964657878366813*x))^(-1)], 
     "AudioSetID" -> "/m/0l14_3", "EntityCanonicalName" -> 
      "BowedStringInstrument::5bddg"|>, "String section" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Bowed string instrument"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Bowed string instrument", "Musical instrument", 
       "Music"}, "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0005267462119480467, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.224067681577774 - 25.779479876516156*x))^(-1)], 
     "AudioSetID" -> "/m/02qmj0d", "EntityCanonicalName" -> 
      "StringSection::kczw7"|>, "Violin, fiddle" -> 
    <|"DepthIndex" -> 0.75, "FirstChildren" -> {"Pizzicato"}, 
     "FirstParents" -> {"Bowed string instrument"}, 
     "FlattenedChildren" -> {"Pizzicato"}, "FlattenedParents" -> 
      {"Bowed string instrument", "Musical instrument", "Music"}, 
     "BottomDepth" -> 1, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.013930583281268255, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.7689147584815545 - 8.483819036027898*x))^(-1)], 
     "AudioSetID" -> "/m/07y_7", "EntityCanonicalName" -> 
      "ViolinFiddle::x776m"|>, "Pizzicato" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Violin, fiddle"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Violin, fiddle", "Bowed string instrument", "Musical instrument", 
       "Music"}, "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0016990994011246057, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.864951053766009 - 11.644082931344487*x))^(-1)], 
     "AudioSetID" -> "/m/0d8_n", "EntityCanonicalName" -> 
      "Pizzicato::857j7"|>, "Cello" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Bowed string instrument"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Bowed string instrument", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.002559570069437037, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.526032589423095 - 9.434873425459653*x))^(-1)], 
     "AudioSetID" -> "/m/01xqw", "EntityCanonicalName" -> "Cello::f8866"|>, 
   "Double bass" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Bowed string instrument"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Bowed string instrument", "Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0010697468875241912, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.44431073844314 - 13.11741370815747*x))^(-1)], 
     "AudioSetID" -> "/m/02fsn", "EntityCanonicalName" -> 
      "DoubleBass::zb958"|>, "Wind instrument, woodwind instrument" -> 
    <|"DepthIndex" -> 0.5, "FirstChildren" -> {"Flute", "Saxophone", 
       "Clarinet", "Oboe", "Bassoon"}, "FirstParents" -> 
      {"Musical instrument"}, "FlattenedChildren" -> 
      {"Flute", "Saxophone", "Clarinet", "Oboe", "Bassoon", "Alto saxophone", 
       "Soprano saxophone"}, "FlattenedParents" -> {"Musical instrument", 
       "Music"}, "BottomDepth" -> 2, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.002753607728997455, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.873895612758234 - 11.6968702384507*x))^(-1)], 
     "AudioSetID" -> "/m/085jw", "EntityCanonicalName" -> 
      "WindInstrumentWoodwindInstrument::2jsqj"|>, 
   "Flute" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Wind instrument, woodwind instrument"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Wind instrument, woodwind instrument", "Musical instrument", 
       "Music"}, "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0022969588914455805, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.406377884587877 - 10.751203961851388*x))^(-1)], 
     "AudioSetID" -> "/m/0l14j_", "EntityCanonicalName" -> "Flute::945x9"|>, 
   "Saxophone" -> <|"DepthIndex" -> 0.75, "FirstChildren" -> 
      {"Alto saxophone", "Soprano saxophone"}, "FirstParents" -> 
      {"Wind instrument, woodwind instrument"}, "FlattenedChildren" -> 
      {"Alto saxophone", "Soprano saxophone"}, "FlattenedParents" -> 
      {"Wind instrument, woodwind instrument", "Musical instrument", 
       "Music"}, "BottomDepth" -> 1, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0014507108788077352, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.818085153934083 - 10.221322593982759*x))^(-1)], 
     "AudioSetID" -> "/m/06ncr", "EntityCanonicalName" -> 
      "Saxophone::zr7pt"|>, "Alto saxophone" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Saxophone"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Saxophone", "Wind instrument, woodwind instrument", 
       "Musical instrument", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 4, 
     "Restrictions" -> {"blacklist"}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/02pprs", "EntityCanonicalName" -> 
      "AltoSaxophone::jnj3g"|>, "Soprano saxophone" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Saxophone"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Saxophone", "Wind instrument, woodwind instrument", 
       "Musical instrument", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 4, 
     "Restrictions" -> {"blacklist"}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/03t22m", "EntityCanonicalName" -> 
      "SopranoSaxophone::jyq45"|>, "Clarinet" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> 
      {"Wind instrument, woodwind instrument"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Wind instrument, woodwind instrument", 
       "Musical instrument", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0009930461372791043, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.591036995419855 - 12.341139901411465*x))^(-1)], 
     "AudioSetID" -> "/m/01wy6", "EntityCanonicalName" -> 
      "Clarinet::z4w2y"|>, "Oboe" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> 
      {"Wind instrument, woodwind instrument"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Wind instrument, woodwind instrument", 
       "Musical instrument", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], "AudioSetID" -> "/m/05kms", 
     "EntityCanonicalName" -> "Oboe::6g3m6"|>, 
   "Bassoon" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Wind instrument, woodwind instrument"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Wind instrument, woodwind instrument", "Musical instrument", 
       "Music"}, "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/01c3q", "EntityCanonicalName" -> "Bassoon::869h3"|>, 
   "Harp" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Musical instrument"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00095494973815075, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.935266947506874 - 9.595393647283752*x))^(-1)], 
     "AudioSetID" -> "/m/03m5k", "EntityCanonicalName" -> "Harp::8978p"|>, 
   "Bell" -> <|"DepthIndex" -> 0.41666666666666663, 
     "FirstChildren" -> {"Church bell", "Cowbell", "Jingle bell", 
       "Bicycle bell", "Tuning fork", "Chime", 
       "Change ringing (campanology)"}, "FirstParents" -> 
      {"Musical instrument", "Sounds of things"}, "FlattenedChildren" -> 
      {"Cowbell", "Church bell", "Jingle bell", "Bicycle bell", 
       "Tuning fork", "Chime", "Change ringing (campanology)", "Wind chime"}, 
     "FlattenedParents" -> {"Musical instrument", "Sounds of things", 
       "Music"}, "BottomDepth" -> 2, "TopDepth" -> 
      <|"Musical instrument" -> 2, "Sounds of things" -> 1|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0008543752444518944, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.210888480880104 - 11.657584303722514*x))^(-1)], 
     "AudioSetID" -> "/m/0395lw", "EntityCanonicalName" -> "Bell::ymz7g"|>, 
   "Church bell" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Bell"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Bell", "Musical instrument", "Sounds of things", 
       "Music"}, "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0005358893477388517, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.933661080655916 - 12.07219666224953*x))^(-1)], 
     "AudioSetID" -> "/m/03w41f", "EntityCanonicalName" -> 
      "ChurchBell::76mps"|>, "Jingle bell" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Bell"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Bell", "Musical instrument", "Sounds of things", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0005196348841107538, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.033236210686523 - 14.87183608098245*x))^(-1)], 
     "AudioSetID" -> "/m/027m70_", "EntityCanonicalName" -> 
      "JingleBell::6n5vd"|>, "Bicycle bell" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Bell", "Bicycle", "Alarm"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Bell", "Bicycle", "Alarm", "Musical instrument", "Sounds of things", 
       "Non-motorized land vehicle", "Sounds of things", "Music", "Vehicle", 
       "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> <|"Bell" -> 4, "Bicycle" -> 4, "Alarm" -> 2|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00005079519883780585, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(11.171231693077834 - 174.135970655725*x))^(-1)], 
     "AudioSetID" -> "/m/0gy1t2s", "EntityCanonicalName" -> 
      "BicycleBell::hb36h"|>, "Tuning fork" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Bell"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Bell", "Musical instrument", "Sounds of things", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.000039620255093488566, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(10.339904914072523 - 50.10980118381533*x))^(-1)], 
     "AudioSetID" -> "/m/07n_g", "EntityCanonicalName" -> 
      "TuningFork::6f677"|>, "Chime" -> <|"DepthIndex" -> 0.75, 
     "FirstChildren" -> {"Wind chime"}, "FirstParents" -> {"Bell"}, 
     "FlattenedChildren" -> {"Wind chime"}, "FlattenedParents" -> 
      {"Bell", "Musical instrument", "Sounds of things", "Music"}, 
     "BottomDepth" -> 1, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0003845196552021903, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.488940164984081 - 20.351166183449475*x))^(-1)], 
     "AudioSetID" -> "/m/0f8s22", "EntityCanonicalName" -> "Chime::8v2s6"|>, 
   "Wind chime" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Chime"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Chime", "Bell", "Musical instrument", 
       "Sounds of things", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 4, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0002229909228979677, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.548966632741806 - 16.553957730970776*x))^(-1)], 
     "AudioSetID" -> "/m/026fgl", "EntityCanonicalName" -> 
      "WindChime::8x877"|>, "Change ringing (campanology)" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> {"Bell"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Bell", "Musical instrument", "Sounds of things", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0002737861217357735, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(9.321633501435006 - 22.150933992830367*x))^(-1)], 
     "AudioSetID" -> "/m/0150b9", "EntityCanonicalName" -> 
      "ChangeRingingCampanology::52p66"|>, 
   "Harmonica" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Musical instrument"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0010443492881052883, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.251744903591073 - 10.584450850271509*x))^(-1)], 
     "AudioSetID" -> "/m/03qjg", "EntityCanonicalName" -> 
      "Harmonica::4f7nn"|>, "Accordion" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Musical instrument"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Musical instrument", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0013760419365161606, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.919411990659174 - 11.092697841851445*x))^(-1)], 
     "AudioSetID" -> "/m/0mkg", "EntityCanonicalName" -> 
      "Accordion::48h5d"|>, "Bagpipes" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Musical instrument"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Musical instrument", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0008274537890678573, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.418457206679934 - 11.479385502724584*x))^(-1)], 
     "AudioSetID" -> "/m/0192l", "EntityCanonicalName" -> 
      "Bagpipes::fjh5b"|>, "Didgeridoo" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Musical instrument"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Musical instrument", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0005567153792623522, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.351096329262129 - 11.728706926028536*x))^(-1)], 
     "AudioSetID" -> "/m/02bxd", "EntityCanonicalName" -> 
      "Didgeridoo::jsrj4"|>, "Shofar" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Musical instrument"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Musical instrument", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00010209834966398976, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.78612208622783 - 31.782222171464976*x))^(-1)], 
     "AudioSetID" -> "/m/0l14l2", "EntityCanonicalName" -> "Shofar::26tb8"|>, 
   "Theremin" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Musical instrument"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0002504203302703828, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.75905924893193 - 33.362577149278984*x))^(-1)], 
     "AudioSetID" -> "/m/07kc_", "EntityCanonicalName" -> 
      "Theremin::n9d29"|>, "Singing bowl" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Musical instrument"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Musical instrument", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00038096399128354386, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.40344727029313 - 21.8866199332577*x))^(-1)], 
     "AudioSetID" -> "/m/0l14t7", "EntityCanonicalName" -> 
      "SingingBowl::k8chv"|>, "Musical ensemble" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Musical instrument"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/05229", "EntityCanonicalName" -> 
      "MusicalEnsemble::6jxt7"|>, "Bass (instrument role)" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Musical instrument"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/01vj9c", "EntityCanonicalName" -> 
      "BassInstrumentRole::tjhn7"|>, "Scratching (performance technique)" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Musical instrument"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Musical instrument", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0013745180805510263, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.628414620795 - 10.162090234141333*x))^(-1)], 
     "AudioSetID" -> "/m/01hgjl", "EntityCanonicalName" -> 
      "ScratchingPerformanceTechnique::pk9wt"|>, 
   "Music genre" -> <|"DepthIndex" -> 0.25, "FirstChildren" -> 
      {"Pop music", "Hip hop music", "Rock music", "Rhythm and blues", 
       "Soul music", "Reggae", "Country", "Funk", "Folk music", 
       "Middle Eastern music", "Jazz", "Disco", "Classical music", 
       "Electronic music", "Music of Latin America", "Blues", 
       "Music for children", "New-age music", "Vocal music", 
       "Music of Africa", "Christian music", "Music of Asia", "Ska", 
       "Traditional music", "Independent music"}, 
     "FirstParents" -> {"Music"}, "FlattenedChildren" -> 
      {"Pop music", "Hip hop music", "Rock music", "Rhythm and blues", 
       "Soul music", "Reggae", "Country", "Funk", "Folk music", 
       "Middle Eastern music", "Jazz", "Disco", "Classical music", 
       "Electronic music", "Music of Latin America", "Blues", 
       "Music for children", "New-age music", "Vocal music", 
       "Music of Africa", "Christian music", "Music of Asia", "Ska", 
       "Traditional music", "Independent music", "Grime music", "Trap music", 
       "Beatboxing", "Heavy metal", "Punk rock", "Grunge", 
       "Progressive rock", "Rock and roll", "Psychedelic rock", "Dub", 
       "Swing music", "Bluegrass", "Opera", "House music", "Techno", 
       "Dubstep", "Electro", "Drum and bass", "Electronica", 
       "Electronic dance music", "Ambient music", "Trance music", 
       "Noise music", "UK garage", "Cumbia", "Salsa music", "Soca music", 
       "Kuduro", "Funk carioca", "Flamenco", "Chant", "Beatboxing", 
       "A capella", "Afrobeat", "Kwaito", "Gospel music", "Carnatic music", 
       "Music of Bollywood", "Oldschool jungle", "Drone music", "Mantra"}, 
     "FlattenedParents" -> {"Music"}, "BottomDepth" -> 3, "TopDepth" -> 1, 
     "Restrictions" -> {"abstract"}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/0kpv1t", "EntityCanonicalName" -> 
      "MusicGenre::b27nx"|>, "Pop music" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Music genre"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music genre", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.004218033311491398, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.156167571353009 - 12.749674400868997*x))^(-1)], 
     "AudioSetID" -> "/m/064t9", "EntityCanonicalName" -> 
      "PopMusic::3fm52"|>, "Hip hop music" -> 
    <|"DepthIndex" -> 0.6666666666666666, "FirstChildren" -> 
      {"Grime music", "Trap music", "Beatboxing"}, 
     "FirstParents" -> {"Music genre"}, "FlattenedChildren" -> 
      {"Grime music", "Trap music", "Beatboxing"}, "FlattenedParents" -> 
      {"Music genre", "Music"}, "BottomDepth" -> 1, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0037039858992528028, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.583798985527898 - 14.296462160341287*x))^(-1)], 
     "AudioSetID" -> "/m/0glt670", "EntityCanonicalName" -> 
      "HipHopMusic::4c965"|>, "Grime music" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Hip hop music"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Hip hop music", "Music genre", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/04j_h4", "EntityCanonicalName" -> 
      "GrimeMusic::8782r"|>, "Trap music" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Hip hop music"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Hip hop music", "Music genre", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/0n8zsc8", "EntityCanonicalName" -> 
      "TrapMusic::d39ph"|>, "Beatboxing" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Hip hop music", 
       "Vocal music"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Hip hop music", "Vocal music", "Music genre", "Music genre", "Music", 
       "Music"}, "BottomDepth" -> 0, "TopDepth" -> <|"Hip hop music" -> 3, 
       "Vocal music" -> 3|>, "Restrictions" -> {}, 
     "ClassPrior" -> 0.001155082821571705, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.762350731251167 - 10.65849658429258*x))^(-1)], 
     "AudioSetID" -> "/m/02cz_7", "EntityCanonicalName" -> 
      "Beatboxing::f272v"|>, "Rock music" -> 
    <|"DepthIndex" -> 0.6666666666666666, "FirstChildren" -> 
      {"Heavy metal", "Punk rock", "Grunge", "Progressive rock", 
       "Rock and roll", "Psychedelic rock"}, "FirstParents" -> 
      {"Music genre"}, "FlattenedChildren" -> {"Heavy metal", "Punk rock", 
       "Grunge", "Progressive rock", "Rock and roll", "Psychedelic rock"}, 
     "FlattenedParents" -> {"Music genre", "Music"}, "BottomDepth" -> 1, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0041296496655136156, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.196988770133495 - 12.666488129515693*x))^(-1)], 
     "AudioSetID" -> "/m/06by7", "EntityCanonicalName" -> 
      "RockMusic::dr3g4"|>, "Heavy metal" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Rock music"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Rock music", "Music genre", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0031406671441415357, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.821934091488126 - 13.281155779947284*x))^(-1)], 
     "AudioSetID" -> "/m/03lty", "EntityCanonicalName" -> 
      "HeavyMetal::392f8"|>, "Punk rock" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Rock music"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Rock music", "Music genre", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0019505356353717446, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.999143903723681 - 23.785013633285494*x))^(-1)], 
     "AudioSetID" -> "/m/05r6t", "EntityCanonicalName" -> 
      "PunkRock::7x8r8"|>, "Grunge" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Rock music"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Rock music", "Music genre", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0009305680427086031, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.646850651293763 - 27.12038037712858*x))^(-1)], 
     "AudioSetID" -> "/m/0dls3", "EntityCanonicalName" -> "Grunge::r8vt4"|>, 
   "Progressive rock" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Rock music"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Rock music", "Music genre", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.001667606377845166, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.013265272259642 - 26.115073408796356*x))^(-1)], 
     "AudioSetID" -> "/m/0dl5d", "EntityCanonicalName" -> 
      "ProgressiveRock::9kb53"|>, "Rock and roll" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Rock music"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Rock music", "Music genre", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.004514677272704184, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.209379889462433 - 14.314094475197088*x))^(-1)], 
     "AudioSetID" -> "/m/07sbbz2", "EntityCanonicalName" -> 
      "RockAndRoll::473sv"|>, "Psychedelic rock" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Rock music"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Rock music", "Music genre", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0010717786954777034, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.442937063684241 - 21.70326327332677*x))^(-1)], 
     "AudioSetID" -> "/m/05w3f", "EntityCanonicalName" -> 
      "PsychedelicRock::x588f"|>, "Rhythm and blues" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Music genre"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music genre", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0024300423124006317, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.5648332874099244 - 12.126375314579402*x))^(-1)], 
     "AudioSetID" -> "/m/06j6l", "EntityCanonicalName" -> 
      "RhythmAndBlues::gf5bb"|>, "Soul music" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Music genre"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music genre", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0013704544646440018, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.024316832413228 - 20.5304941096296*x))^(-1)], 
     "AudioSetID" -> "/m/0gywn", "EntityCanonicalName" -> 
      "SoulMusic::8nr7j"|>, "Reggae" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Dub"}, "FirstParents" -> {"Music genre"}, 
     "FlattenedChildren" -> {"Dub"}, "FlattenedParents" -> 
      {"Music genre", "Music"}, "BottomDepth" -> 1, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.001527411629052822, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.363037161989662 - 16.534534148740125*x))^(-1)], 
     "AudioSetID" -> "/m/06cqb", "EntityCanonicalName" -> "Reggae::597z7"|>, 
   "Dub" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Reggae"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Reggae", "Music genre", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/0190y4", "EntityCanonicalName" -> "Dub::pz3tx"|>, 
   "Country" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Swing music", "Bluegrass"}, 
     "FirstParents" -> {"Music genre"}, "FlattenedChildren" -> 
      {"Swing music", "Bluegrass"}, "FlattenedParents" -> 
      {"Music genre", "Music"}, "BottomDepth" -> 1, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0026870660185199296, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.483335439003019 - 14.559073042644407*x))^(-1)], 
     "AudioSetID" -> "/m/01lyv", "EntityCanonicalName" -> "Country::7w597"|>, 
   "Swing music" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Country"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Country", "Music genre", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0007502450868343924, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.690539658309972 - 18.67316013182142*x))^(-1)], 
     "AudioSetID" -> "/m/015y_n", "EntityCanonicalName" -> 
      "SwingMusic::858y3"|>, "Bluegrass" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Country"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Country", "Music genre", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0013008650422362078, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.3679565640579305 - 11.395309708774567*x))^(-1)], 
     "AudioSetID" -> "/m/0gg8l", "EntityCanonicalName" -> 
      "Bluegrass::wxb43"|>, "Funk" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Music genre"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music genre", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.001926661891917976, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.64917647715788 - 23.99197917976333*x))^(-1)], 
     "AudioSetID" -> "/m/02x8m", "EntityCanonicalName" -> "Funk::zx63s"|>, 
   "Folk music" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Music genre"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Music genre", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0010118403608490925, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.575800329143588 - 20.812658750532368*x))^(-1)], 
     "AudioSetID" -> "/m/02w4v", "EntityCanonicalName" -> 
      "FolkMusic::9s6z9"|>, "Middle Eastern music" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Music genre"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music genre", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0009361555145807618, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.44969190852733 - 19.899336481175617*x))^(-1)], 
     "AudioSetID" -> "/m/06j64v", "EntityCanonicalName" -> 
      "MiddleEasternMusic::35pj3"|>, "Jazz" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Music genre"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music genre", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0023838186814582286, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.40275666763864 - 10.238119125856667*x))^(-1)], 
     "AudioSetID" -> "/m/03_d0", "EntityCanonicalName" -> "Jazz::6dybb"|>, 
   "Disco" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Music genre"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Music genre", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0019525674433252568, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.922357471132016 - 11.126837820194421*x))^(-1)], 
     "AudioSetID" -> "/m/026z9", "EntityCanonicalName" -> "Disco::xc238"|>, 
   "Classical music" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Opera"}, "FirstParents" -> {"Music genre"}, 
     "FlattenedChildren" -> {"Opera"}, "FlattenedParents" -> 
      {"Music genre", "Music"}, "BottomDepth" -> 1, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.003373817106807065, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.498483969878525 - 14.17837540041606*x))^(-1)], 
     "AudioSetID" -> "/m/0ggq0m", "EntityCanonicalName" -> 
      "ClassicalMusic::b4q9x"|>, "Opera" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Classical music"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Classical music", "Music genre", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.001067715079570679, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.39640082555505 - 10.52565023651896*x))^(-1)], 
     "AudioSetID" -> "/m/05lls", "EntityCanonicalName" -> "Opera::2gnrt"|>, 
   "Electronic music" -> <|"DepthIndex" -> 0.5, "FirstChildren" -> 
      {"House music", "Techno", "Dubstep", "Electro", "Drum and bass", 
       "Electronica", "Electronic dance music", "Ambient music", 
       "Trance music", "Noise music", "UK garage"}, 
     "FirstParents" -> {"Music genre"}, "FlattenedChildren" -> 
      {"House music", "Techno", "Dubstep", "Electro", "Drum and bass", 
       "Electronica", "Electronic dance music", "Ambient music", 
       "Trance music", "Noise music", "UK garage", "Oldschool jungle", 
       "Drone music"}, "FlattenedParents" -> {"Music genre", "Music"}, 
     "BottomDepth" -> 2, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.019074613067572854, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(5.691606005796206 - 8.154045065368958*x))^(-1)], 
     "AudioSetID" -> "/m/02lkt", "EntityCanonicalName" -> 
      "ElectronicMusic::6frkk"|>, "House music" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Electronic music"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Electronic music", "Music genre", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.003153873895839365, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.538702185037423 - 20.341573633245492*x))^(-1)], 
     "AudioSetID" -> "/m/03mb9", "EntityCanonicalName" -> 
      "HouseMusic::m5983"|>, "Techno" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Electronic music"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Electronic music", "Music genre", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.008456892654506295, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.039756807245583 - 10.051307283910374*x))^(-1)], 
     "AudioSetID" -> "/m/07gxw", "EntityCanonicalName" -> "Techno::tx2sk"|>, 
   "Dubstep" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Electronic music"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Electronic music", "Music genre", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.009073546368397259, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.529186991518017 - 8.385593173050546*x))^(-1)], 
     "AudioSetID" -> "/m/07s72n", "EntityCanonicalName" -> 
      "Dubstep::8rskt"|>, "Electro" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Electronic music"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Electronic music", "Music genre", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/029h7y", "EntityCanonicalName" -> 
      "Electro::2s7g8"|>, "Drum and bass" -> <|"DepthIndex" -> 0.75, 
     "FirstChildren" -> {"Oldschool jungle"}, "FirstParents" -> 
      {"Electronic music"}, "FlattenedChildren" -> {"Oldschool jungle"}, 
     "FlattenedParents" -> {"Electronic music", "Music genre", "Music"}, 
     "BottomDepth" -> 1, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0017742762954045584, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.337817257636998 - 11.20958025671439*x))^(-1)], 
     "AudioSetID" -> "/m/0283d", "EntityCanonicalName" -> 
      "DrumAndBass::k2mh6"|>, "Oldschool jungle" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Drum and bass"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Drum and bass", "Electronic music", 
       "Music genre", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 4, 
     "Restrictions" -> {"blacklist"}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/01f9gb", "EntityCanonicalName" -> 
      "OldschoolJungle::qt7kr"|>, "Electronica" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Electronic music"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Electronic music", "Music genre", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0026032539404375497, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.815443804479209 - 47.05639294816676*x))^(-1)], 
     "AudioSetID" -> "/m/0m0jc", "EntityCanonicalName" -> 
      "Electronica::274y3"|>, "Electronic dance music" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Electronic music"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Electronic music", "Music genre", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0020551737449776247, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.007913757000283 - 26.145492401517174*x))^(-1)], 
     "AudioSetID" -> "/m/08cyft", "EntityCanonicalName" -> 
      "ElectronicDanceMusic::4ht7s"|>, "Ambient music" -> 
    <|"DepthIndex" -> 0.75, "FirstChildren" -> {"Drone music"}, 
     "FirstParents" -> {"Electronic music"}, "FlattenedChildren" -> 
      {"Drone music"}, "FlattenedParents" -> {"Electronic music", 
       "Music genre", "Music"}, "BottomDepth" -> 1, "TopDepth" -> 3, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0013303262575621352, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.546089496632173 - 14.095847710102126*x))^(-1)], 
     "AudioSetID" -> "/m/0fd3y", "EntityCanonicalName" -> 
      "AmbientMusic::hst4f"|>, "Drone music" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Ambient music"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Ambient music", "Electronic music", "Music genre", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/052smk", "EntityCanonicalName" -> 
      "DroneMusic::8qdsm"|>, "Trance music" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Electronic music"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Electronic music", "Music genre", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0021384778710716265, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.967930632824664 - 15.20203974124317*x))^(-1)], 
     "AudioSetID" -> "/m/07lnk", "EntityCanonicalName" -> 
      "TranceMusic::8xd9g"|>, "Noise music" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Electronic music"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Electronic music", "Music genre", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/0m0fw", "EntityCanonicalName" -> 
      "NoiseMusic::35ms3"|>, "UK garage" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Electronic music"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Electronic music", "Music genre", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/0bmfpc", "EntityCanonicalName" -> 
      "UKGarage::w34vb"|>, "Music of Latin America" -> 
    <|"DepthIndex" -> 0.6666666666666666, "FirstChildren" -> 
      {"Cumbia", "Salsa music", "Soca music", "Kuduro", "Funk carioca", 
       "Flamenco"}, "FirstParents" -> {"Music genre"}, 
     "FlattenedChildren" -> {"Cumbia", "Salsa music", "Soca music", "Kuduro", 
       "Funk carioca", "Flamenco"}, "FlattenedParents" -> 
      {"Music genre", "Music"}, "BottomDepth" -> 1, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.001997267218302526, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.943015580644661 - 11.208822961513619*x))^(-1)], 
     "AudioSetID" -> "/m/0g293", "EntityCanonicalName" -> 
      "MusicOfLatinAmerica::cx3zg"|>, 
   "Cumbia" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Music of Latin America"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Music of Latin America", "Music genre", 
       "Music"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {"blacklist"}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/02ccj9", "EntityCanonicalName" -> "Cumbia::hypg5"|>, 
   "Salsa music" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Music of Latin America"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Music of Latin America", "Music genre", 
       "Music"}, "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0011962269326303279, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.029832498768148 - 9.990837262914972*x))^(-1)], 
     "AudioSetID" -> "/m/0ln16", "EntityCanonicalName" -> 
      "SalsaMusic::875y8"|>, "Soca music" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Music of Latin America"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music of Latin America", "Music genre", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/0kpck", "EntityCanonicalName" -> 
      "SocaMusic::574sr"|>, "Kuduro" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Music of Latin America"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music of Latin America", "Music genre", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/080nby", "EntityCanonicalName" -> "Kuduro::4szj3"|>, 
   "Funk carioca" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Music of Latin America"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Music of Latin America", "Music genre", 
       "Music"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {"blacklist"}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/05q7ms", "EntityCanonicalName" -> 
      "FunkCarioca::3htmb"|>, "Flamenco" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Music of Latin America"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music of Latin America", "Music genre", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0011169864224433506, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.7301103305967285 - 9.605351857663635*x))^(-1)], 
     "AudioSetID" -> "/m/0326g", "EntityCanonicalName" -> 
      "Flamenco::dgrd5"|>, "Blues" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Music genre"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music genre", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.002308641787178276, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.52473819994626 - 21.035624302750325*x))^(-1)], 
     "AudioSetID" -> "/m/0155w", "EntityCanonicalName" -> "Blues::2645d"|>, 
   "Music for children" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Music genre"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Music genre", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0014959186057733824, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.365057308585465 - 9.320547832625708*x))^(-1)], 
     "AudioSetID" -> "/m/05fw6t", "EntityCanonicalName" -> 
      "MusicForChildren::4w482"|>, "New-age music" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Music genre"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music genre", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0014786482381685282, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.054058115249359 - 19.668458755290615*x))^(-1)], 
     "AudioSetID" -> "/m/02v2lh", "EntityCanonicalName" -> 
      "New-ageMusic::z77w4"|>, "Vocal music" -> <|"DepthIndex" -> 0.5, 
     "FirstChildren" -> {"A capella", "Chant", "Beatboxing"}, 
     "FirstParents" -> {"Music genre"}, "FlattenedChildren" -> 
      {"Chant", "Beatboxing", "A capella", "Mantra"}, 
     "FlattenedParents" -> {"Music genre", "Music"}, "BottomDepth" -> 2, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0017519264079159237, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.908499996648079 - 14.106914647973115*x))^(-1)], 
     "AudioSetID" -> "/m/0y4f8", "EntityCanonicalName" -> 
      "VocalMusic::6tt7f"|>, "A capella" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Vocal music"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Vocal music", "Music genre", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0008929795955686269, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.928472579265274 - 14.85922735287026*x))^(-1)], 
     "AudioSetID" -> "/m/0z9c", "EntityCanonicalName" -> "ACapella::p6z6v"|>, 
   "Music of Africa" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Afrobeat", "Kwaito"}, "FirstParents" -> 
      {"Music genre"}, "FlattenedChildren" -> {"Afrobeat", "Kwaito"}, 
     "FlattenedParents" -> {"Music genre", "Music"}, "BottomDepth" -> 1, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0013176274578526838, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.452215076714817 - 21.33743171722819*x))^(-1)], 
     "AudioSetID" -> "/m/0164x2", "EntityCanonicalName" -> 
      "MusicOfAfrica::d87bc"|>, "Afrobeat" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Music of Africa"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music of Africa", "Music genre", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0012439744195378653, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.571476438470163 - 15.538854038753522*x))^(-1)], 
     "AudioSetID" -> "/m/0145m", "EntityCanonicalName" -> 
      "Afrobeat::2hmq8"|>, "Kwaito" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Music of Africa"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music of Africa", "Music genre", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/022dgg", "EntityCanonicalName" -> "Kwaito::z936r"|>, 
   "Christian music" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Gospel music"}, "FirstParents" -> {"Music genre"}, 
     "FlattenedChildren" -> {"Gospel music"}, "FlattenedParents" -> 
      {"Music genre", "Music"}, "BottomDepth" -> 1, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.001475092574249882, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.065661859503329 - 18.0343433547545*x))^(-1)], 
     "AudioSetID" -> "/m/02mscn", "EntityCanonicalName" -> 
      "ChristianMusic::52qf7"|>, "Gospel music" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Christian music"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Christian music", "Music genre", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.002067364592698698, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.800668848687245 - 13.928248503355295*x))^(-1)], 
     "AudioSetID" -> "/m/016cjb", "EntityCanonicalName" -> 
      "GospelMusic::4h66z"|>, "Music of Asia" -> 
    <|"DepthIndex" -> 0.6666666666666666, "FirstChildren" -> 
      {"Carnatic music", "Music of Bollywood"}, "FirstParents" -> 
      {"Music genre"}, "FlattenedChildren" -> {"Carnatic music", 
       "Music of Bollywood"}, "FlattenedParents" -> {"Music genre", "Music"}, 
     "BottomDepth" -> 1, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0018037375107304858, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.953585773885948 - 9.21405519751629*x))^(-1)], 
     "AudioSetID" -> "/m/028sqc", "EntityCanonicalName" -> 
      "MusicOfAsia::6ts28"|>, "Carnatic music" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Music of Asia"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Music of Asia", "Music genre", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0007573564146716853, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.892040210453642 - 13.9783163813225*x))^(-1)], 
     "AudioSetID" -> "/m/015vgc", "EntityCanonicalName" -> 
      "CarnaticMusic::q5wf8"|>, "Music of Bollywood" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Music of Asia"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Music of Asia", "Music genre", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0010565401358263618, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.143665418465098 - 10.762899369646036*x))^(-1)], 
     "AudioSetID" -> "/m/0dq0md", "EntityCanonicalName" -> 
      "MusicOfBollywood::48y48"|>, "Ska" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Music genre"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music genre", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0007802142541486978, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.599319376199592 - 21.064428292006724*x))^(-1)], 
     "AudioSetID" -> "/m/06rqw", "EntityCanonicalName" -> "Ska::848r7"|>, 
   "Traditional music" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Music genre"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Music genre", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0007949448618116615, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.564569825851908 - 35.83946873643537*x))^(-1)], 
     "AudioSetID" -> "/m/02p0sh1", "EntityCanonicalName" -> 
      "TraditionalMusic::9r5c8"|>, "Independent music" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Music genre"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music genre", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0014182019515515394, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.924947087600022 - 23.679359199540844*x))^(-1)], 
     "AudioSetID" -> "/m/05rwpb", "EntityCanonicalName" -> 
      "IndependentMusic::zsn7q"|>, "Musical concepts" -> 
    <|"DepthIndex" -> 0.3333333333333333, "FirstChildren" -> 
      {"Song", "Melody", "Musical note", "Beat", "Chord", "Harmony", 
       "Bassline", "Loop", "Drone"}, "FirstParents" -> {"Music"}, 
     "FlattenedChildren" -> {"Song", "Melody", "Musical note", "Beat", 
       "Chord", "Harmony", "Bassline", "Loop", "Drone", "Drum beat"}, 
     "FlattenedParents" -> {"Music"}, "BottomDepth" -> 2, "TopDepth" -> 1, 
     "Restrictions" -> {"abstract", "blacklist"}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/t/dd00027", "EntityCanonicalName" -> 
      "MusicalConcepts::sq4c4"|>, "Song" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Musical concepts"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Musical concepts", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {"blacklist"}, "ClassPrior" -> 0.0008325333089516379, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.835040232532211 - 127.51773412389366*x))^(-1)], 
     "AudioSetID" -> "/m/074ft", "EntityCanonicalName" -> "Song::386wf"|>, 
   "Melody" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Musical concepts"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Musical concepts", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/09dsr", "EntityCanonicalName" -> "Melody::86ywh"|>, 
   "Musical note" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Musical concepts"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Musical concepts", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/05jcn", "EntityCanonicalName" -> 
      "MusicalNote::v4634"|>, "Beat" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Drum beat"}, "FirstParents" -> 
      {"Musical concepts"}, "FlattenedChildren" -> {"Drum beat"}, 
     "FlattenedParents" -> {"Musical concepts", "Music"}, "BottomDepth" -> 1, 
     "TopDepth" -> 2, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/022c7z", "EntityCanonicalName" -> "Beat::824r6"|>, 
   "Drum beat" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Beat"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Beat", "Musical concepts", "Music"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/05xp3j", "EntityCanonicalName" -> 
      "DrumBeat::5xm79"|>, "Chord" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Musical concepts"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Musical concepts", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {"blacklist"}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/01gp74", "EntityCanonicalName" -> "Chord::zw35y"|>, 
   "Harmony" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Musical concepts"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Musical concepts", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/0b128", "EntityCanonicalName" -> "Harmony::pjtd5"|>, 
   "Bassline" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Musical concepts"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Musical concepts", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/021wwz", "EntityCanonicalName" -> 
      "Bassline::3c269"|>, "Loop" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Musical concepts"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Musical concepts", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {"blacklist"}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], "AudioSetID" -> "/m/0kc2j", 
     "EntityCanonicalName" -> "Loop::9gjz9"|>, 
   "Drone" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Musical concepts"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Musical concepts", "Music"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/03w6d1", "EntityCanonicalName" -> "Drone::43z6p"|>, 
   "Music role" -> <|"DepthIndex" -> 0.5, "FirstChildren" -> 
      {"Background music", "Theme music", "Jingle (music)", 
       "Soundtrack music", "Lullaby", "Video game music", "Christmas music", 
       "Dance music", "Wedding music", "Birthday music"}, 
     "FirstParents" -> {"Music"}, "FlattenedChildren" -> 
      {"Background music", "Theme music", "Jingle (music)", 
       "Soundtrack music", "Lullaby", "Video game music", "Christmas music", 
       "Dance music", "Wedding music", "Birthday music"}, 
     "FlattenedParents" -> {"Music"}, "BottomDepth" -> 1, "TopDepth" -> 1, 
     "Restrictions" -> {"abstract"}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/t/dd00028", "EntityCanonicalName" -> 
      "MusicRole::6q93j"|>, "Background music" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Music role"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music role", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0023482620422717645, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.381786455050657 - 21.22110318567092*x))^(-1)], 
     "AudioSetID" -> "/m/025td0t", "EntityCanonicalName" -> 
      "BackgroundMusic::f7qjk"|>, "Theme music" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Music role"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music role", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0020439988012333076, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.562820397185866 - 21.647027870063177*x))^(-1)], 
     "AudioSetID" -> "/m/02cjck", "EntityCanonicalName" -> 
      "ThemeMusic::96xk9"|>, "Jingle (music)" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Music role"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music role", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0006369717934260853, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.770563523016511 - 34.16047361289848*x))^(-1)], 
     "AudioSetID" -> "/m/03r5q_", "EntityCanonicalName" -> 
      "JingleMusic::s2925"|>, "Soundtrack music" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Music role"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music role", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.003162001127653414, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.1274841745661615 - 24.188655547213504*x))^(-1)], 
     "AudioSetID" -> "/m/0l14gg", "EntityCanonicalName" -> 
      "SoundtrackMusic::84q4h"|>, "Lullaby" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Music role"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music role", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0015248718691109316, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.818177342242236 - 23.80166917683063*x))^(-1)], 
     "AudioSetID" -> "/m/07pkxdp", "EntityCanonicalName" -> 
      "Lullaby::8rxcd"|>, "Video game music" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Music role"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music role", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0015690636920998226, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.97141552924473 - 19.36365135026905*x))^(-1)], 
     "AudioSetID" -> "/m/01z7dr", "EntityCanonicalName" -> 
      "VideoGameMusic::d7439"|>, "Christmas music" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Music role"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music role", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0009869507134185676, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.03998962064662 - 26.69470968684922*x))^(-1)], 
     "AudioSetID" -> "/m/0140xf", "EntityCanonicalName" -> 
      "ChristmasMusic::t22x5"|>, "Dance music" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Music role"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music role", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.002308641787178276, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.637301999113352 - 18.58616571661937*x))^(-1)], 
     "AudioSetID" -> "/m/0ggx5q", "EntityCanonicalName" -> 
      "DanceMusic::2s4nm"|>, "Wedding music" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Music role"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music role", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0003428675921551895, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.236041531622467 - 97.65458990904443*x))^(-1)], 
     "AudioSetID" -> "/m/04wptg", "EntityCanonicalName" -> 
      "WeddingMusic::m9724"|>, "Birthday music" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Music role"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music role", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/t/dd00029", "EntityCanonicalName" -> 
      "BirthdayMusic::z263f"|>, "Music mood" -> <|"DepthIndex" -> 0.5, 
     "FirstChildren" -> {"Happy music", "Funny music", "Sad music", 
       "Tender music", "Exciting music", "Angry music", "Scary music"}, 
     "FirstParents" -> {"Music"}, "FlattenedChildren" -> 
      {"Happy music", "Funny music", "Sad music", "Tender music", 
       "Exciting music", "Angry music", "Scary music"}, 
     "FlattenedParents" -> {"Music"}, "BottomDepth" -> 1, "TopDepth" -> 1, 
     "Restrictions" -> {"abstract"}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/t/dd00030", "EntityCanonicalName" -> 
      "MusicMood::35qpg"|>, "Happy music" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Music mood"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music mood", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0006527183050658051, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.9733904025775715 - 89.44939140124288*x))^(-1)], 
     "AudioSetID" -> "/t/dd00031", "EntityCanonicalName" -> 
      "HappyMusic::m8973"|>, "Funny music" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Music mood"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music mood", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0005028724684942779, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.904110573228206 - 78.32539182779801*x))^(-1)], 
     "AudioSetID" -> "/t/dd00032", "EntityCanonicalName" -> 
      "FunnyMusic::58c5f"|>, "Sad music" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Music mood"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music mood", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0008035800456140886, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.6567868067373235 - 47.33353937315184*x))^(-1)], 
     "AudioSetID" -> "/t/dd00033", "EntityCanonicalName" -> 
      "SadMusic::xnw8g"|>, "Tender music" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Music mood"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music mood", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0019388527396390492, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.885784138329687 - 29.64785450232239*x))^(-1)], 
     "AudioSetID" -> "/t/dd00034", "EntityCanonicalName" -> 
      "TenderMusic::k78k4"|>, "Exciting music" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Music mood"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music mood", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0027058602420899176, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.69199980216612 - 61.2602754271335*x))^(-1)], 
     "AudioSetID" -> "/t/dd00035", "EntityCanonicalName" -> 
      "ExcitingMusic::3qmd8"|>, "Angry music" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Music mood"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music mood", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00048407824492428976, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.998483247434501 - 20.182548519214706*x))^(-1)], 
     "AudioSetID" -> "/t/dd00036", "EntityCanonicalName" -> 
      "AngryMusic::27p4r"|>, "Scary music" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Music mood"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Music mood", "Music"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0007771665422184295, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.989353365414723 - 13.406099780072568*x))^(-1)], 
     "AudioSetID" -> "/t/dd00037", "EntityCanonicalName" -> 
      "ScaryMusic::39y6c"|>, "Natural sounds" -> <|"DepthIndex" -> 0., 
     "FirstChildren" -> {"Wind", "Thunderstorm", "Water", "Fire"}, 
     "FirstParents" -> {}, "FlattenedChildren" -> {"Wind", "Thunderstorm", 
       "Water", "Fire", "Howl (wind)", "Rustling leaves", 
       "Wind noise (microphone)", "Thunder", "Rain", "Stream", "Waterfall", 
       "Ocean", "Steam", "Gurgling", "Crackle", "Wildfire", "Raindrop", 
       "Rain on surface", "Waves, surf", "Hiss"}, "FlattenedParents" -> {}, 
     "BottomDepth" -> 3, "TopDepth" -> 0, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/059j3w", "EntityCanonicalName" -> 
      "NaturalSounds::4338k"|>, "Wind" -> <|"DepthIndex" -> 0.5, 
     "FirstChildren" -> {"Howl (wind)", "Rustling leaves", 
       "Wind noise (microphone)"}, "FirstParents" -> {"Natural sounds"}, 
     "FlattenedChildren" -> {"Howl (wind)", "Rustling leaves", 
       "Wind noise (microphone)"}, "FlattenedParents" -> {"Natural sounds"}, 
     "BottomDepth" -> 1, "TopDepth" -> 1, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0032046690946771713, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.774879688073209 - 12.44298554655148*x))^(-1)], 
     "AudioSetID" -> "/m/03m9d0z", "EntityCanonicalName" -> "Wind::226m5"|>, 
   "Howl (wind)" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Wind"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Wind", "Natural sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/07q8f3b", "EntityCanonicalName" -> 
      "HowlWind::c55k2"|>, "Rustling leaves" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Wind"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Wind", "Natural sounds"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0007974846217535519, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.514984481738492 - 32.53971468259728*x))^(-1)], 
     "AudioSetID" -> "/m/09t49", "EntityCanonicalName" -> 
      "RustlingLeaves::pz45w"|>, "Wind noise (microphone)" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Wind", "Microphone"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Wind", "Microphone", "Natural sounds", 
       "Sound equipment", "Miscellaneous sources", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> <|"Wind" -> 2, "Microphone" -> 4|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0030715856737221197, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.865228414907585 - 13.49709051842262*x))^(-1)], 
     "AudioSetID" -> "/t/dd00092", "EntityCanonicalName" -> 
      "WindNoiseMicrophone::8d766"|>, "Thunderstorm" -> 
    <|"DepthIndex" -> 0.5, "FirstChildren" -> {"Thunder"}, 
     "FirstParents" -> {"Natural sounds"}, "FlattenedChildren" -> 
      {"Thunder"}, "FlattenedParents" -> {"Natural sounds"}, 
     "BottomDepth" -> 1, "TopDepth" -> 1, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0005597630911926205, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.016107702130643 - 13.88697914123035*x))^(-1)], 
     "AudioSetID" -> "/m/0jb2l", "EntityCanonicalName" -> 
      "Thunderstorm::xxc8b"|>, "Thunder" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Thunderstorm"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Thunderstorm", "Natural sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.000572461890902072, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.040425639597052 - 16.829036369375654*x))^(-1)], 
     "AudioSetID" -> "/m/0ngt1", "EntityCanonicalName" -> "Thunder::t9323"|>, 
   "Water" -> <|"DepthIndex" -> 0.3333333333333333, 
     "FirstChildren" -> {"Rain", "Stream", "Waterfall", "Ocean", "Steam", 
       "Gurgling"}, "FirstParents" -> {"Natural sounds"}, 
     "FlattenedChildren" -> {"Rain", "Stream", "Waterfall", "Ocean", "Steam", 
       "Gurgling", "Raindrop", "Rain on surface", "Waves, surf", "Hiss"}, 
     "FlattenedParents" -> {"Natural sounds"}, "BottomDepth" -> 2, 
     "TopDepth" -> 1, "Restrictions" -> {}, "ClassPrior" -> 
      0.004061076147082578, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.576366493978882 - 11.055197104773566*x))^(-1)], 
     "AudioSetID" -> "/m/0838f", "EntityCanonicalName" -> "Water::6269h"|>, 
   "Rain" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Raindrop", "Rain on surface"}, 
     "FirstParents" -> {"Water"}, "FlattenedChildren" -> 
      {"Raindrop", "Rain on surface"}, "FlattenedParents" -> 
      {"Water", "Natural sounds"}, "BottomDepth" -> 1, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0015695716440882008, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.46496722140717 - 11.932754095662368*x))^(-1)], 
     "AudioSetID" -> "/m/06mb1", "EntityCanonicalName" -> "Rain::g65d2"|>, 
   "Raindrop" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Rain"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Rain", "Water", "Natural sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0005455404355180348, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.929374018601143 - 22.273588370074965*x))^(-1)], 
     "AudioSetID" -> "/m/07r10fb", "EntityCanonicalName" -> 
      "Raindrop::qdy3m"|>, "Rain on surface" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Rain"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Rain", "Water", "Natural sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0013592795208996846, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.17863562163422 - 12.780971686157601*x))^(-1)], 
     "AudioSetID" -> "/t/dd00038", "EntityCanonicalName" -> 
      "RainOnSurface::54dgm"|>, "Stream" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Water"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Water", "Natural sounds"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0013572477129461724, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.259156082520555 - 17.913837382637272*x))^(-1)], 
     "AudioSetID" -> "/m/0j6m2", "EntityCanonicalName" -> "Stream::5678w"|>, 
   "Waterfall" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Water"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Water", "Natural sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0006847192803336228, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.006950366890567 - 23.70862783040467*x))^(-1)], 
     "AudioSetID" -> "/m/0j2kx", "EntityCanonicalName" -> 
      "Waterfall::dq9dn"|>, "Ocean" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Waves, surf"}, "FirstParents" -> {"Water"}, 
     "FlattenedChildren" -> {"Waves, surf"}, "FlattenedParents" -> 
      {"Water", "Natural sounds"}, "BottomDepth" -> 1, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0015909056276000792, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.333641895340591 - 10.811205076553108*x))^(-1)], 
     "AudioSetID" -> "/m/05kq4", "EntityCanonicalName" -> "Ocean::g56g6"|>, 
   "Waves, surf" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Ocean"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Ocean", "Water", "Natural sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0013404852973296964, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.617714155662285 - 11.516459653921702*x))^(-1)], 
     "AudioSetID" -> "/m/034srq", "EntityCanonicalName" -> 
      "WavesSurf::8pz7f"|>, "Steam" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Hiss"}, "FirstParents" -> {"Water"}, 
     "FlattenedChildren" -> {"Hiss"}, "FlattenedParents" -> 
      {"Water", "Natural sounds"}, "BottomDepth" -> 1, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0009950779452326166, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.539757516584724 - 14.554104852911466*x))^(-1)], 
     "AudioSetID" -> "/m/06wzb", "EntityCanonicalName" -> "Steam::rp2vf"|>, 
   "Gurgling" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Water"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Water", "Natural sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0009011068273826758, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.457044749484279 - 35.73886147245307*x))^(-1)], 
     "AudioSetID" -> "/m/07swgks", "EntityCanonicalName" -> 
      "Gurgling::crg39"|>, "Fire" -> <|"DepthIndex" -> 0.5, 
     "FirstChildren" -> {"Crackle", "Wildfire"}, "FirstParents" -> 
      {"Natural sounds"}, "FlattenedChildren" -> {"Crackle", "Wildfire"}, 
     "FlattenedParents" -> {"Natural sounds"}, "BottomDepth" -> 1, 
     "TopDepth" -> 1, "Restrictions" -> {}, "ClassPrior" -> 
      0.0006501785451239149, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.719847581979295 - 57.66849865844201*x))^(-1)], 
     "AudioSetID" -> "/m/02_41", "EntityCanonicalName" -> "Fire::qm5d9"|>, 
   "Crackle" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Fire", "Onomatopoeia"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Fire", "Onomatopoeia", "Natural sounds", 
       "Source-ambiguous sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> <|"Fire" -> 2, "Onomatopoeia" -> 2|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0006710045766474153, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.6560214246829545 - 69.84325833318925*x))^(-1)], 
     "AudioSetID" -> "/m/07pzfmf", "EntityCanonicalName" -> 
      "Crackle::knn9m"|>, "Wildfire" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Fire"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Fire", "Natural sounds"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], "AudioSetID" -> "/m/0fjy1", 
     "EntityCanonicalName" -> "Wildfire::44drc"|>, 
   "Sounds of things" -> <|"DepthIndex" -> 0., "FirstChildren" -> 
      {"Vehicle", "Engine", "Domestic sounds, home sounds", "Bell", "Alarm", 
       "Mechanisms", "Tools", "Explosion", "Wood", "Glass", "Liquid", 
       "Miscellaneous sources", "Specific impact sounds"}, 
     "FirstParents" -> {}, "FlattenedChildren" -> {"Bell", "Vehicle", 
       "Engine", "Domestic sounds, home sounds", "Alarm", "Mechanisms", 
       "Tools", "Explosion", "Wood", "Glass", "Liquid", 
       "Miscellaneous sources", "Specific impact sounds", "Cowbell", 
       "Church bell", "Jingle bell", "Bicycle bell", "Tuning fork", "Chime", 
       "Change ringing (campanology)", "Boat, Water vehicle", 
       "Motor vehicle (road)", "Rail transport", "Aircraft", 
       "Non-motorized land vehicle", "Jet engine", 
       "Light engine (high frequency)", "Medium engine (mid frequency)", 
       "Heavy engine (low frequency)", "Engine knocking", "Engine starting", 
       "Idling", "Accelerating, revving, vroom", "Door", 
       "Cupboard open or close", "Drawer open or close", 
       "Dishes, pots, and pans", "Cutlery, silverware", "Chopping (food)", 
       "Frying (food)", "Microwave oven", "Blender", "Kettle whistle", 
       "Water tap, faucet", "Sink (filling or washing)", 
       "Bathtub (filling or washing)", "Hair dryer", "Toilet flush", 
       "Toothbrush", "Vacuum cleaner", "Zipper (clothing)", 
       "Velcro, hook and loop fastener", "Keys jangling", "Coin (dropping)", 
       "Packing tape, duct tape", "Scissors", 
       "Electric shaver, electric razor", "Shuffling cards", "Typing", 
       "Writing", "Bicycle bell", "Vehicle horn, car horn, honking", 
       "Car alarm", "Air horn, truck horn", "Doorbell", "Telephone", 
       "Alarm clock", "Siren", "Buzzer", "Smoke detector, smoke alarm", 
       "Fire alarm", "Foghorn", "Whistle", "Ratchet, pawl", "Clock", "Gears", 
       "Pulleys", "Sewing machine", "Mechanical fan", "Air conditioning", 
       "Cash register", "Printer", "Camera", "Hammer", "Jackhammer", 
       "Sawing", "Filing (rasp)", "Sanding", "Power tool", 
       "Gunshot, gunfire", "Fireworks", "Burst, pop", "Eruption", "Boom", 
       "Chop", "Splinter", "Crack", "Snap", "Chink, clink", "Shatter", 
       "Splash, splatter", "Squish", "Drip", "Pour", "Fill (with liquid)", 
       "Spray", "Pump (liquid)", "Stir", "Boiling", "Sonar", 
       "Duck call (hunting tool)", "Arrow", "Sound equipment", 
       "Basketball bounce", "Wind chime", "Sailboat, sailing ship", 
       "Rowboat, canoe, kayak", "Motorboat, speedboat", "Ship", "Car", 
       "Truck", "Bus", "Emergency vehicle", "Motorcycle", 
       "Traffic noise, roadway noise", "Train", "Railroad car, train wagon", 
       "Train wheels squealing", "Subway, metro, underground", 
       "Aircraft engine", "Helicopter", "Fixed-wing aircraft, airplane", 
       "Bicycle", "Skateboard", "Dental drill, dentist's drill", 
       "Lawn mower", "Chainsaw", "Doorbell", "Sliding door", "Slam", "Knock", 
       "Tap", "Squeak", "Electric toothbrush", "Typewriter", 
       "Computer keyboard", "Toot", "Ding-dong", "Telephone bell ringing", 
       "Ringtone", "Cellphone buzz, vibrating alert", 
       "Telephone dialing, DTMF", "Dial tone", "Busy signal", 
       "Police car (siren)", "Ambulance (siren)", 
       "Fire engine, fire truck (siren)", "Civil defense siren", 
       "Kettle whistle", "Steam whistle", "Tick", "Tick-tock", 
       "Single-lens reflex camera", "Drill", "Machine gun", "Fusillade", 
       "Artillery fire", "Cap gun", "Firecracker", "Sonic boom", "Slosh", 
       "Trickle, dribble", "Gush", "Whoosh, swoosh, swish", "Thump, thud", 
       "Wobble", "Microphone", "Electronic tuner", "Guitar amplifier", 
       "Effects unit", "Vehicle horn, car horn, honking", "Car alarm", 
       "Power windows, electric windows", "Skidding", "Tire squeal", 
       "Car passing by", "Race car, auto racing", "Air brake", 
       "Air horn, truck horn", "Reversing beeps", 
       "Ice cream truck, ice cream van", "Police car (siren)", 
       "Ambulance (siren)", "Fire engine, fire truck (siren)", 
       "Train whistle", "Train horn", "Jet engine", "Propeller, airscrew", 
       "Bicycle bell", "Ding-dong", "Dental drill, dentist's drill", "Thunk", 
       "Clunk", "Wind noise (microphone)", "Chorus effect", "Toot"}, 
     "FlattenedParents" -> {}, "BottomDepth" -> 5, "TopDepth" -> 0, 
     "Restrictions" -> {"blacklist"}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/t/dd00041", "EntityCanonicalName" -> 
      "SoundsOfThings::6gnbb"|>, "Vehicle" -> <|"DepthIndex" -> 0.2, 
     "FirstChildren" -> {"Boat, Water vehicle", "Motor vehicle (road)", 
       "Rail transport", "Aircraft", "Non-motorized land vehicle"}, 
     "FirstParents" -> {"Sounds of things"}, "FlattenedChildren" -> 
      {"Boat, Water vehicle", "Motor vehicle (road)", "Rail transport", 
       "Aircraft", "Non-motorized land vehicle", "Sailboat, sailing ship", 
       "Rowboat, canoe, kayak", "Motorboat, speedboat", "Ship", "Car", 
       "Truck", "Bus", "Emergency vehicle", "Motorcycle", 
       "Traffic noise, roadway noise", "Train", "Railroad car, train wagon", 
       "Train wheels squealing", "Subway, metro, underground", 
       "Aircraft engine", "Helicopter", "Fixed-wing aircraft, airplane", 
       "Bicycle", "Skateboard", "Vehicle horn, car horn, honking", 
       "Car alarm", "Power windows, electric windows", "Skidding", 
       "Tire squeal", "Car passing by", "Race car, auto racing", "Air brake", 
       "Air horn, truck horn", "Reversing beeps", 
       "Ice cream truck, ice cream van", "Police car (siren)", 
       "Ambulance (siren)", "Fire engine, fire truck (siren)", 
       "Train whistle", "Train horn", "Jet engine", "Propeller, airscrew", 
       "Bicycle bell", "Toot"}, "FlattenedParents" -> {"Sounds of things"}, 
     "BottomDepth" -> 4, "TopDepth" -> 1, "Restrictions" -> {}, 
     "ClassPrior" -> 0.06274172165246941, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(5.944155397121946 - 7.7662295190578545*x))^(-1)], 
     "AudioSetID" -> "/m/07yv9", "EntityCanonicalName" -> "Vehicle::s4n6z"|>, 
   "Boat, Water vehicle" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Sailboat, sailing ship", "Rowboat, canoe, kayak", 
       "Motorboat, speedboat", "Ship"}, "FirstParents" -> {"Vehicle"}, 
     "FlattenedChildren" -> {"Sailboat, sailing ship", 
       "Rowboat, canoe, kayak", "Motorboat, speedboat", "Ship"}, 
     "FlattenedParents" -> {"Vehicle", "Sounds of things"}, 
     "BottomDepth" -> 1, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.006544961370251284, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(5.941115298248616 - 10.934125897071258*x))^(-1)], 
     "AudioSetID" -> "/m/019jd", "EntityCanonicalName" -> 
      "BoatWaterVehicle::9ghjh"|>, "Sailboat, sailing ship" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Boat, Water vehicle"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Boat, Water vehicle", "Vehicle", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0012815628666778417, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.130214046704018 - 15.673297809437734*x))^(-1)], 
     "AudioSetID" -> "/m/0hsrw", "EntityCanonicalName" -> 
      "SailboatSailingShip::8gwq3"|>, "Rowboat, canoe, kayak" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Boat, Water vehicle"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Boat, Water vehicle", "Vehicle", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0013491204811321234, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.467903235935847 - 11.29664078437008*x))^(-1)], 
     "AudioSetID" -> "/m/056ks2", "EntityCanonicalName" -> 
      "RowboatCanoeKayak::hwn2t"|>, "Motorboat, speedboat" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Boat, Water vehicle"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Boat, Water vehicle", "Vehicle", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {}, "ClassPrior" -> 0.003982343588883979, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.3581927569227625 - 11.309091758962202*x))^(-1)], 
     "AudioSetID" -> "/m/02rlv9", "EntityCanonicalName" -> 
      "MotorboatSpeedboat::79g85"|>, "Ship" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Boat, Water vehicle"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Boat, Water vehicle", "Vehicle", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00036928109555084854, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.418902376953737 - 69.35424403599477*x))^(-1)], 
     "AudioSetID" -> "/m/06q74", "EntityCanonicalName" -> "Ship::2tj3w"|>, 
   "Motor vehicle (road)" -> <|"DepthIndex" -> 0.4, 
     "FirstChildren" -> {"Car", "Truck", "Bus", "Emergency vehicle", 
       "Motorcycle", "Traffic noise, roadway noise"}, 
     "FirstParents" -> {"Vehicle"}, "FlattenedChildren" -> 
      {"Car", "Truck", "Bus", "Emergency vehicle", "Motorcycle", 
       "Traffic noise, roadway noise", "Vehicle horn, car horn, honking", 
       "Car alarm", "Power windows, electric windows", "Skidding", 
       "Tire squeal", "Car passing by", "Race car, auto racing", "Air brake", 
       "Air horn, truck horn", "Reversing beeps", 
       "Ice cream truck, ice cream van", "Police car (siren)", 
       "Ambulance (siren)", "Fire engine, fire truck (siren)", "Toot"}, 
     "FlattenedParents" -> {"Vehicle", "Sounds of things"}, 
     "BottomDepth" -> 3, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.004414102779005329, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.368953134193223 - 25.864776819249833*x))^(-1)], 
     "AudioSetID" -> "/m/012f08", "EntityCanonicalName" -> 
      "MotorVehicleRoad::zjj5h"|>, 
   "Car" -> <|"DepthIndex" -> 0.6000000000000001, "FirstChildren" -> 
      {"Vehicle horn, car horn, honking", "Car alarm", 
       "Power windows, electric windows", "Skidding", "Tire squeal", 
       "Car passing by", "Race car, auto racing"}, 
     "FirstParents" -> {"Motor vehicle (road)"}, "FlattenedChildren" -> 
      {"Vehicle horn, car horn, honking", "Car alarm", 
       "Power windows, electric windows", "Skidding", "Tire squeal", 
       "Car passing by", "Race car, auto racing", "Toot"}, 
     "FlattenedParents" -> {"Motor vehicle (road)", "Vehicle", 
       "Sounds of things"}, "BottomDepth" -> 2, "TopDepth" -> 3, 
     "Restrictions" -> {}, "ClassPrior" -> 0.020284554703889388, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(5.611423182780556 - 8.964168356924285*x))^(-1)], 
     "AudioSetID" -> "/m/0k4j", "EntityCanonicalName" -> "Car::5d24j"|>, 
   "Vehicle horn, car horn, honking" -> <|"DepthIndex" -> 0.7333333333333334, 
     "FirstChildren" -> {"Toot"}, "FirstParents" -> {"Car", "Alarm"}, 
     "FlattenedChildren" -> {"Toot"}, "FlattenedParents" -> 
      {"Car", "Alarm", "Motor vehicle (road)", "Sounds of things", "Vehicle", 
       "Sounds of things"}, "BottomDepth" -> 1, 
     "TopDepth" -> <|"Car" -> 4, "Alarm" -> 2|>, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0017707206314859119, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.252098160419602 - 11.101938585094585*x))^(-1)], 
     "AudioSetID" -> "/m/0912c9", "EntityCanonicalName" -> 
      "VehicleHornCarHornHonking::j4ftp"|>, 
   "Toot" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Vehicle horn, car horn, honking"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Vehicle horn, car horn, honking", "Car", "Alarm", 
       "Motor vehicle (road)", "Sounds of things", "Vehicle", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 5, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00031239047285250597, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.592993806562871 - 39.5524896484971*x))^(-1)], 
     "AudioSetID" -> "/m/07qv_d5", "EntityCanonicalName" -> "Toot::875p9"|>, 
   "Car alarm" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Car", "Alarm"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Car", "Alarm", "Motor vehicle (road)", 
       "Sounds of things", "Vehicle", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> <|"Car" -> 4, "Alarm" -> 2|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0001970853714906867, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.207802260769366 - 13.416347062695651*x))^(-1)], 
     "AudioSetID" -> "/m/02mfyn", "EntityCanonicalName" -> 
      "CarAlarm::nn9q2"|>, "Power windows, electric windows" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> {"Car"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Car", "Motor vehicle (road)", "Vehicle", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00007263713433806236, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.68568363238535 - 267.6084798898726*x))^(-1)], 
     "AudioSetID" -> "/m/04gxbd", "EntityCanonicalName" -> 
      "PowerWindowsElectricWindows::wrs2r"|>, 
   "Skidding" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Car"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Car", "Motor vehicle (road)", "Vehicle", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 4, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0006989419360082085, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.77936786955349 - 10.411067106335194*x))^(-1)], 
     "AudioSetID" -> "/m/07rknqz", "EntityCanonicalName" -> 
      "Skidding::b9xh6"|>, "Tire squeal" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Car"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Car", "Motor vehicle (road)", "Vehicle", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0007091009757757697, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.912549934748833 - 17.334177396072057*x))^(-1)], 
     "AudioSetID" -> "/m/0h9mv", "EntityCanonicalName" -> 
      "TireSqueal::382y3"|>, "Car passing by" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Car"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Car", "Motor vehicle (road)", "Vehicle", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0018174522144166934, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.052008742154126 - 24.101260006544226*x))^(-1)], 
     "AudioSetID" -> "/t/dd00134", "EntityCanonicalName" -> 
      "CarPassingBy::kk2kr"|>, "Race car, auto racing" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> {"Car"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Car", "Motor vehicle (road)", "Vehicle", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0032011134307585246, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.115958658871324 - 9.694172051772098*x))^(-1)], 
     "AudioSetID" -> "/m/0ltv", "EntityCanonicalName" -> 
      "RaceCarAutoRacing::2rn45"|>, "Truck" -> <|"DepthIndex" -> 0.75, 
     "FirstChildren" -> {"Air brake", "Air horn, truck horn", 
       "Reversing beeps", "Ice cream truck, ice cream van"}, 
     "FirstParents" -> {"Motor vehicle (road)"}, "FlattenedChildren" -> 
      {"Air brake", "Air horn, truck horn", "Reversing beeps", 
       "Ice cream truck, ice cream van"}, "FlattenedParents" -> 
      {"Motor vehicle (road)", "Vehicle", "Sounds of things"}, 
     "BottomDepth" -> 1, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0054046091563425425, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.1089770247312165 - 12.176533752122152*x))^(-1)], 
     "AudioSetID" -> "/m/07r04", "EntityCanonicalName" -> "Truck::d98hj"|>, 
   "Air brake" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Truck"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Truck", "Motor vehicle (road)", "Vehicle", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 4, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0003758844713997633, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.122418252175589 - 49.962960565311676*x))^(-1)], 
     "AudioSetID" -> "/m/0gvgw0", "EntityCanonicalName" -> 
      "AirBrake::z6b8g"|>, "Air horn, truck horn" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Truck", "Alarm"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Truck", "Alarm", "Motor vehicle (road)", 
       "Sounds of things", "Vehicle", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> <|"Truck" -> 4, "Alarm" -> 2|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00034185168817843335, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.361557428096763 - 19.332937092011576*x))^(-1)], 
     "AudioSetID" -> "/m/05x_td", "EntityCanonicalName" -> 
      "AirHornTruckHorn::6vd7t"|>, "Reversing beeps" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> {"Truck"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Truck", "Motor vehicle (road)", "Vehicle", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00013054366101316104, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.536758945584214 - 16.497350570362272*x))^(-1)], 
     "AudioSetID" -> "/m/02rhddq", "EntityCanonicalName" -> 
      "ReversingBeeps::79n48"|>, "Ice cream truck, ice cream van" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> {"Truck"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Truck", "Motor vehicle (road)", "Vehicle", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00007974846217535518, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.692320303945486 - 19.228824080080262*x))^(-1)], 
     "AudioSetID" -> "/m/03cl9h", "EntityCanonicalName" -> 
      "IceCreamTruckIceCreamVan::6d333"|>, 
   "Bus" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Motor vehicle (road)"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Motor vehicle (road)", "Vehicle", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {}, "ClassPrior" -> 0.002561093925402171, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.83112826749013 - 10.730499255878794*x))^(-1)], 
     "AudioSetID" -> "/m/01bjv", "EntityCanonicalName" -> "Bus::8whpn"|>, 
   "Emergency vehicle" -> <|"DepthIndex" -> 0.75, "FirstChildren" -> 
      {"Police car (siren)", "Ambulance (siren)", 
       "Fire engine, fire truck (siren)"}, "FirstParents" -> 
      {"Motor vehicle (road)"}, "FlattenedChildren" -> 
      {"Police car (siren)", "Ambulance (siren)", 
       "Fire engine, fire truck (siren)"}, "FlattenedParents" -> 
      {"Motor vehicle (road)", "Vehicle", "Sounds of things"}, 
     "BottomDepth" -> 1, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0026951932503339786, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.9484844982329745 - 12.387680320506744*x))^(-1)], 
     "AudioSetID" -> "/m/03j1ly", "EntityCanonicalName" -> 
      "EmergencyVehicle::475b9"|>, "Police car (siren)" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Emergency vehicle", "Siren"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Emergency vehicle", "Siren", 
       "Motor vehicle (road)", "Alarm", "Vehicle", "Sounds of things", 
       "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> <|"Emergency vehicle" -> 4, "Siren" -> 3|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0017504025519507896, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.863351676603022 - 15.41466789059575*x))^(-1)], 
     "AudioSetID" -> "/m/04qvtq", "EntityCanonicalName" -> 
      "PoliceCarSiren::4fjx4"|>, "Ambulance (siren)" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Emergency vehicle", "Siren"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Emergency vehicle", "Siren", 
       "Motor vehicle (road)", "Alarm", "Vehicle", "Sounds of things", 
       "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> <|"Emergency vehicle" -> 4, "Siren" -> 3|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0008899318836383585, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.243120879047506 - 16.155711678345597*x))^(-1)], 
     "AudioSetID" -> "/m/012n7d", "EntityCanonicalName" -> 
      "AmbulanceSiren::djg59"|>, "Fire engine, fire truck (siren)" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Emergency vehicle", "Siren"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Emergency vehicle", "Siren", 
       "Motor vehicle (road)", "Alarm", "Vehicle", "Sounds of things", 
       "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> <|"Emergency vehicle" -> 4, "Siren" -> 3|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0015350309088784928, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.856999018168116 - 13.011326008713622*x))^(-1)], 
     "AudioSetID" -> "/m/012ndj", "EntityCanonicalName" -> 
      "FireEngineFireTruckSiren::73dc5"|>, "Motorcycle" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Motor vehicle (road)"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Motor vehicle (road)", "Vehicle", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0035317901751926407, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.528605047755674 - 11.019521685090579*x))^(-1)], 
     "AudioSetID" -> "/m/04_sv", "EntityCanonicalName" -> 
      "Motorcycle::29j85"|>, "Traffic noise, roadway noise" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Motor vehicle (road)"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Motor vehicle (road)", "Vehicle", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0007629438865438438, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.656690086375575 - 24.189413973637556*x))^(-1)], 
     "AudioSetID" -> "/m/0btp2", "EntityCanonicalName" -> 
      "TrafficNoiseRoadwayNoise::rsm4r"|>, "Rail transport" -> 
    <|"DepthIndex" -> 0.5, "FirstChildren" -> 
      {"Train", "Railroad car, train wagon", "Train wheels squealing", 
       "Subway, metro, underground"}, "FirstParents" -> {"Vehicle"}, 
     "FlattenedChildren" -> {"Train", "Railroad car, train wagon", 
       "Train wheels squealing", "Subway, metro, underground", 
       "Train whistle", "Train horn"}, "FlattenedParents" -> 
      {"Vehicle", "Sounds of things"}, "BottomDepth" -> 2, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0043353702208067295, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.0871253022153144 - 8.48774991863674*x))^(-1)], 
     "AudioSetID" -> "/m/06d_3", "EntityCanonicalName" -> 
      "RailTransport::p7qdf"|>, "Train" -> <|"DepthIndex" -> 0.75, 
     "FirstChildren" -> {"Train whistle", "Train horn"}, 
     "FirstParents" -> {"Rail transport"}, "FlattenedChildren" -> 
      {"Train whistle", "Train horn"}, "FlattenedParents" -> 
      {"Rail transport", "Vehicle", "Sounds of things"}, "BottomDepth" -> 1, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.006203617634061229, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.713530548013175 - 7.6653774819500695*x))^(-1)], 
     "AudioSetID" -> "/m/07jdr", "EntityCanonicalName" -> "Train::4q45f"|>, 
   "Train whistle" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Train"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Train", "Rail transport", "Vehicle", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 4, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00031848589671304266, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.684281842833613 - 28.091426212368155*x))^(-1)], 
     "AudioSetID" -> "/m/04zmvq", "EntityCanonicalName" -> 
      "TrainWhistle::3378m"|>, "Train horn" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Train"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Train", "Rail transport", "Vehicle", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 4, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0010616196557101422, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.139602826317539 - 14.299729107918244*x))^(-1)], 
     "AudioSetID" -> "/m/0284vy3", "EntityCanonicalName" -> 
      "TrainHorn::r23sj"|>, "Railroad car, train wagon" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Rail transport"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Rail transport", "Vehicle", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.003991994676663162, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.232420052284485 - 8.421479931492755*x))^(-1)], 
     "AudioSetID" -> "/m/01g50p", "EntityCanonicalName" -> 
      "RailroadCarTrainWagon::t4t8h"|>, "Train wheels squealing" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Rail transport"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Rail transport", "Vehicle", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00019200585160690612, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.318420506602457 - 48.90284971386934*x))^(-1)], 
     "AudioSetID" -> "/t/dd00048", "EntityCanonicalName" -> 
      "TrainWheelsSquealing::9w57w"|>, "Subway, metro, underground" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Rail transport"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Rail transport", "Vehicle", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0008152629413467839, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.820010079534196 - 11.89245648270497*x))^(-1)], 
     "AudioSetID" -> "/m/0195fx", "EntityCanonicalName" -> 
      "SubwayMetroUnderground::247ns"|>, "Aircraft" -> 
    <|"DepthIndex" -> 0.5, "FirstChildren" -> {"Aircraft engine", 
       "Helicopter", "Fixed-wing aircraft, airplane"}, 
     "FirstParents" -> {"Vehicle"}, "FlattenedChildren" -> 
      {"Aircraft engine", "Helicopter", "Fixed-wing aircraft, airplane", 
       "Jet engine", "Propeller, airscrew"}, "FlattenedParents" -> 
      {"Vehicle", "Sounds of things"}, "BottomDepth" -> 2, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.002620524308042404, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.803279973351892 - 12.023625629402439*x))^(-1)], 
     "AudioSetID" -> "/m/0k5j", "EntityCanonicalName" -> "Aircraft::3twyk"|>, 
   "Aircraft engine" -> <|"DepthIndex" -> 0.75, "FirstChildren" -> 
      {"Jet engine", "Propeller, airscrew"}, "FirstParents" -> {"Aircraft"}, 
     "FlattenedChildren" -> {"Jet engine", "Propeller, airscrew"}, 
     "FlattenedParents" -> {"Aircraft", "Vehicle", "Sounds of things"}, 
     "BottomDepth" -> 1, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0005170951241688635, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.64378041120572 - 18.320305019389618*x))^(-1)], 
     "AudioSetID" -> "/m/014yck", "EntityCanonicalName" -> 
      "AircraftEngine::zwq7c"|>, "Jet engine" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Aircraft engine", "Engine"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Aircraft engine", "Engine", "Aircraft", "Sounds of things", 
       "Vehicle", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> <|"Aircraft engine" -> 4, "Engine" -> 2|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0003108666168873718, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.751388410950907 - 22.771399568018595*x))^(-1)], 
     "AudioSetID" -> "/m/04229", "EntityCanonicalName" -> 
      "JetEngine::cp9nt"|>, "Propeller, airscrew" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Aircraft engine"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Aircraft engine", "Aircraft", "Vehicle", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 4, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00046122040544727713, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.075859047885004 - 36.81404121297231*x))^(-1)], 
     "AudioSetID" -> "/m/02l6bg", "EntityCanonicalName" -> 
      "PropellerAirscrew::sv8r6"|>, "Helicopter" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Aircraft"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Aircraft", "Vehicle", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0017737683434161803, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.32488695733679 - 10.502115381218891*x))^(-1)], 
     "AudioSetID" -> "/m/09ct_", "EntityCanonicalName" -> 
      "Helicopter::v62s7"|>, "Fixed-wing aircraft, airplane" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Aircraft"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Aircraft", "Vehicle", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0014227735194469418, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.276676294639062 - 12.543939396097494*x))^(-1)], 
     "AudioSetID" -> "/m/0cmf2", "EntityCanonicalName" -> 
      "Fixed-wingAircraftAirplane::9twf3"|>, "Non-motorized land vehicle" -> 
    <|"DepthIndex" -> 0.5, "FirstChildren" -> {"Bicycle", "Skateboard"}, 
     "FirstParents" -> {"Vehicle"}, "FlattenedChildren" -> 
      {"Bicycle", "Skateboard", "Bicycle bell"}, "FlattenedParents" -> 
      {"Vehicle", "Sounds of things"}, "BottomDepth" -> 2, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/t/dd00061", "EntityCanonicalName" -> 
      "Non-motorizedLandVehicle::5jxhz"|>, 
   "Bicycle" -> <|"DepthIndex" -> 0.75, "FirstChildren" -> {"Bicycle bell"}, 
     "FirstParents" -> {"Non-motorized land vehicle"}, 
     "FlattenedChildren" -> {"Bicycle bell"}, "FlattenedParents" -> 
      {"Non-motorized land vehicle", "Vehicle", "Sounds of things"}, 
     "BottomDepth" -> 1, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.001397375920028039, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.8344697453928385 - 41.57658024075234*x))^(-1)], 
     "AudioSetID" -> "/m/0199g", "EntityCanonicalName" -> "Bicycle::28dkc"|>, 
   "Skateboard" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Non-motorized land vehicle"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Non-motorized land vehicle", "Vehicle", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0014507108788077352, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.423658123236493 - 10.386926344250508*x))^(-1)], 
     "AudioSetID" -> "/m/06_fw", "EntityCanonicalName" -> 
      "Skateboard::36rs3"|>, "Engine" -> 
    <|"DepthIndex" -> 0.3333333333333333, "FirstChildren" -> 
      {"Light engine (high frequency)", "Medium engine (mid frequency)", 
       "Heavy engine (low frequency)", "Jet engine", "Engine knocking", 
       "Engine starting", "Idling", "Accelerating, revving, vroom"}, 
     "FirstParents" -> {"Sounds of things"}, "FlattenedChildren" -> 
      {"Jet engine", "Light engine (high frequency)", 
       "Medium engine (mid frequency)", "Heavy engine (low frequency)", 
       "Engine knocking", "Engine starting", "Idling", 
       "Accelerating, revving, vroom", "Dental drill, dentist's drill", 
       "Lawn mower", "Chainsaw"}, "FlattenedParents" -> {"Sounds of things"}, 
     "BottomDepth" -> 2, "TopDepth" -> 1, "Restrictions" -> {}, 
     "ClassPrior" -> 0.007917955594837175, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(5.8059770463499865 - 11.45383038888661*x))^(-1)], 
     "AudioSetID" -> "/m/02mk9", "EntityCanonicalName" -> "Engine::96bxx"|>, 
   "Light engine (high frequency)" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Dental drill, dentist's drill", "Lawn mower", 
       "Chainsaw"}, "FirstParents" -> {"Engine"}, "FlattenedChildren" -> 
      {"Dental drill, dentist's drill", "Lawn mower", "Chainsaw"}, 
     "FlattenedParents" -> {"Engine", "Sounds of things"}, 
     "BottomDepth" -> 1, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00015644921242044201, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.701944473602929 - 73.56041110167494*x))^(-1)], 
     "AudioSetID" -> "/t/dd00065", "EntityCanonicalName" -> 
      "LightEngineHighFrequency::bqj3k"|>, "Dental drill, dentist's drill" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Light engine (high frequency)", "Drill"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Light engine (high frequency)", "Drill", 
       "Engine", "Power tool", "Sounds of things", "Tools", 
       "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> <|"Light engine (high frequency)" -> 3, "Drill" -> 4|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00003098507129106157, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(11.562285360764893 - 221.87103311775613*x))^(-1)], 
     "AudioSetID" -> "/m/08j51y", "EntityCanonicalName" -> 
      "DentalDrillDentistsDrill::46z52"|>, "Lawn mower" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Light engine (high frequency)"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Light engine (high frequency)", "Engine", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0007426258070087216, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.733411307227135 - 20.692053159171575*x))^(-1)], 
     "AudioSetID" -> "/m/01yg9g", "EntityCanonicalName" -> 
      "LawnMower::n9r79"|>, "Chainsaw" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> 
      {"Light engine (high frequency)"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Light engine (high frequency)", "Engine", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0008208504132189425, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.343539943059582 - 12.22632588189214*x))^(-1)], 
     "AudioSetID" -> "/m/01j4z9", "EntityCanonicalName" -> 
      "Chainsaw::4zfsc"|>, "Medium engine (mid frequency)" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Engine"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Engine", "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0030025042033027037, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.569425543495018 - 21.705222085454217*x))^(-1)], 
     "AudioSetID" -> "/t/dd00066", "EntityCanonicalName" -> 
      "MediumEngineMidFrequency::283fd"|>, "Heavy engine (low frequency)" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Engine"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Engine", "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0020429828972565512, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.073958888713623 - 65.9160805206653*x))^(-1)], 
     "AudioSetID" -> "/t/dd00067", "EntityCanonicalName" -> 
      "HeavyEngineLowFrequency::bymv4"|>, "Engine knocking" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Engine"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Engine", "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {"blacklist"}, "ClassPrior" -> 0.00029664396121278615, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.70336767998961 - 20.396283140065183*x))^(-1)], 
     "AudioSetID" -> "/m/01h82_", "EntityCanonicalName" -> 
      "EngineKnocking::77h32"|>, "Engine starting" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Engine"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Engine", "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0004911895727615826, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.932479450092197 - 54.41637496721195*x))^(-1)], 
     "AudioSetID" -> "/t/dd00130", "EntityCanonicalName" -> 
      "EngineStarting::876np"|>, "Idling" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Engine"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Engine", "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0030329813226053874, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.874024500001039 - 14.366366152222659*x))^(-1)], 
     "AudioSetID" -> "/m/07pb8fc", "EntityCanonicalName" -> 
      "Idling::7k7z5"|>, "Accelerating, revving, vroom" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Engine"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Engine", "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.005836368346463892, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.342663576507042 - 17.237113314841178*x))^(-1)], 
     "AudioSetID" -> "/m/07q2z82", "EntityCanonicalName" -> 
      "AcceleratingRevvingVroom::vz5zb"|>, "Domestic sounds, home sounds" -> 
    <|"DepthIndex" -> 0.25, "FirstChildren" -> 
      {"Door", "Cupboard open or close", "Drawer open or close", 
       "Dishes, pots, and pans", "Cutlery, silverware", "Chopping (food)", 
       "Frying (food)", "Microwave oven", "Blender", "Kettle whistle", 
       "Water tap, faucet", "Sink (filling or washing)", 
       "Bathtub (filling or washing)", "Hair dryer", "Toilet flush", 
       "Toothbrush", "Vacuum cleaner", "Zipper (clothing)", 
       "Velcro, hook and loop fastener", "Keys jangling", "Coin (dropping)", 
       "Packing tape, duct tape", "Scissors", 
       "Electric shaver, electric razor", "Shuffling cards", "Typing", 
       "Writing"}, "FirstParents" -> {"Sounds of things"}, 
     "FlattenedChildren" -> {"Door", "Cupboard open or close", 
       "Drawer open or close", "Dishes, pots, and pans", 
       "Cutlery, silverware", "Chopping (food)", "Frying (food)", 
       "Microwave oven", "Blender", "Kettle whistle", "Water tap, faucet", 
       "Sink (filling or washing)", "Bathtub (filling or washing)", 
       "Hair dryer", "Toilet flush", "Toothbrush", "Vacuum cleaner", 
       "Zipper (clothing)", "Velcro, hook and loop fastener", 
       "Keys jangling", "Coin (dropping)", "Packing tape, duct tape", 
       "Scissors", "Electric shaver, electric razor", "Shuffling cards", 
       "Typing", "Writing", "Doorbell", "Sliding door", "Slam", "Knock", 
       "Tap", "Squeak", "Electric toothbrush", "Typewriter", 
       "Computer keyboard", "Ding-dong"}, "FlattenedParents" -> 
      {"Sounds of things"}, "BottomDepth" -> 3, "TopDepth" -> 1, 
     "Restrictions" -> {"blacklist"}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/t/dd00071", "EntityCanonicalName" -> 
      "DomesticSoundsHomeSounds::7r839"|>, 
   "Door" -> <|"DepthIndex" -> 0.5, "FirstChildren" -> 
      {"Doorbell", "Sliding door", "Slam", "Knock", "Tap", "Squeak"}, 
     "FirstParents" -> {"Domestic sounds, home sounds"}, 
     "FlattenedChildren" -> {"Doorbell", "Sliding door", "Slam", "Knock", 
       "Tap", "Squeak", "Ding-dong"}, "FlattenedParents" -> 
      {"Domestic sounds, home sounds", "Sounds of things"}, 
     "BottomDepth" -> 2, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0011820042769557421, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.1972784769067255 - 25.164632439283288*x))^(-1)], 
     "AudioSetID" -> "/m/02dgv", "EntityCanonicalName" -> "Door::725gc"|>, 
   "Doorbell" -> <|"DepthIndex" -> 0.7083333333333333, 
     "FirstChildren" -> {"Ding-dong"}, "FirstParents" -> {"Door", "Alarm"}, 
     "FlattenedChildren" -> {"Ding-dong"}, "FlattenedParents" -> 
      {"Door", "Alarm", "Domestic sounds, home sounds", "Sounds of things", 
       "Sounds of things"}, "BottomDepth" -> 1, 
     "TopDepth" -> <|"Door" -> 3, "Alarm" -> 2|>, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00010311425364074587, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.492708763279152 - 39.56959499309535*x))^(-1)], 
     "AudioSetID" -> "/m/03wwcy", "EntityCanonicalName" -> 
      "Doorbell::4htf5"|>, "Ding-dong" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Doorbell"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Doorbell", "Door", "Alarm", "Domestic sounds, home sounds", 
       "Sounds of things", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 4, "Restrictions" -> {}, "ClassPrior" -> 
      0.0001711798200834057, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(9.198878079227844 - 70.09702809966585*x))^(-1)], 
     "AudioSetID" -> "/m/07r67yg", "EntityCanonicalName" -> 
      "Ding-dong::p7vtb"|>, "Sliding door" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Door"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Door", "Domestic sounds, home sounds", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00045969654948214294, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.820682006773313 - 42.835029783617784*x))^(-1)], 
     "AudioSetID" -> "/m/02y_763", "EntityCanonicalName" -> 
      "SlidingDoor::yv5b3"|>, "Slam" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Door"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Door", "Domestic sounds, home sounds", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0003570902478297751, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.338507111039473 - 77.2389349343166*x))^(-1)], 
     "AudioSetID" -> "/m/07rjzl8", "EntityCanonicalName" -> "Slam::wrs6p"|>, 
   "Knock" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Door", "Generic impact sounds"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Door", "Generic impact sounds", "Domestic sounds, home sounds", 
       "Source-ambiguous sounds", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> <|"Door" -> 3, "Generic impact sounds" -> 2|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0000980347337569653, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.397525829651856 - 49.40465132295683*x))^(-1)], 
     "AudioSetID" -> "/m/07r4wb8", "EntityCanonicalName" -> "Knock::7y32m"|>, 
   "Tap" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Door", "Generic impact sounds"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Door", "Generic impact sounds", "Domestic sounds, home sounds", 
       "Source-ambiguous sounds", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> <|"Door" -> 3, "Generic impact sounds" -> 2|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00163255769064708, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.254268145945086 - 11.805987148808416*x))^(-1)], 
     "AudioSetID" -> "/m/07qcpgn", "EntityCanonicalName" -> "Tap::c7669"|>, 
   "Squeak" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Door", "Brief tone"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Door", "Brief tone", 
       "Domestic sounds, home sounds", "Onomatopoeia", "Sounds of things", 
       "Source-ambiguous sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> <|"Door" -> 3, "Brief tone" -> 3|>, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00003606459117484215, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(11.263810698585193 - 836.4646369097164*x))^(-1)], 
     "AudioSetID" -> "/m/07q6cd_", "EntityCanonicalName" -> 
      "Squeak::7s84k"|>, "Cupboard open or close" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Domestic sounds, home sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00003454073520970798, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(10.60642957286768 - 1936.84331173583*x))^(-1)], 
     "AudioSetID" -> "/m/0642b4", "EntityCanonicalName" -> 
      "CupboardOpenOrClose::vdrp2"|>, "Drawer open or close" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Domestic sounds, home sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00007568484626833072, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.964938241454899 - 275.87015151480784*x))^(-1)], 
     "AudioSetID" -> "/m/0fqfqc", "EntityCanonicalName" -> 
      "DrawerOpenOrClose::sjr3w"|>, "Dishes, pots, and pans" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Domestic sounds, home sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0008355810208819063, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.646046732232888 - 18.83849474336733*x))^(-1)], 
     "AudioSetID" -> "/m/04brg2", "EntityCanonicalName" -> 
      "DishesPotsAndPans::947p6"|>, "Cutlery, silverware" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Domestic sounds, home sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00037080495151598273, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.223129987365711 - 25.039474396185284*x))^(-1)], 
     "AudioSetID" -> "/m/023pjk", "EntityCanonicalName" -> 
      "CutlerySilverware::w99j8"|>, "Chopping (food)" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Domestic sounds, home sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00004114411105862274, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.993735848245912 - 810.8198973379496*x))^(-1)], 
     "AudioSetID" -> "/m/07pn_8q", "EntityCanonicalName" -> 
      "ChoppingFood::957z2"|>, "Frying (food)" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Domestic sounds, home sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.000728403151334136, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.477141775601641 - 9.736121246837651*x))^(-1)], 
     "AudioSetID" -> "/m/0dxrf", "EntityCanonicalName" -> 
      "FryingFood::x3t3z"|>, "Microwave oven" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Domestic sounds, home sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0003479471120389701, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.330855207579159 - 66.81416343761092*x))^(-1)], 
     "AudioSetID" -> "/m/0fx9l", "EntityCanonicalName" -> 
      "MicrowaveOven::yjxv6"|>, "Blender" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Domestic sounds, home sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0005653505630647791, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.769117831119964 - 20.07158354434126*x))^(-1)], 
     "AudioSetID" -> "/m/02pjr4", "EntityCanonicalName" -> 
      "Blender::92w7f"|>, "Kettle whistle" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> 
      {"Domestic sounds, home sounds", "Whistle"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Domestic sounds, home sounds", "Whistle", 
       "Sounds of things", "Alarm", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> <|"Domestic sounds, home sounds" -> 2, "Whistle" -> 3|>, 
     "Restrictions" -> {}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/g/11b630rrvh", "EntityCanonicalName" -> 
      "KettleWhistle::62scz"|>, "Water tap, faucet" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Domestic sounds, home sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0011042876227338992, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.245221663409525 - 11.454711583964189*x))^(-1)], 
     "AudioSetID" -> "/m/02jz0l", "EntityCanonicalName" -> 
      "WaterTapFaucet::kph67"|>, "Sink (filling or washing)" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Domestic sounds, home sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0006887828962406473, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.479172742055846 - 11.294290417206208*x))^(-1)], 
     "AudioSetID" -> "/m/0130jx", "EntityCanonicalName" -> 
      "SinkFillingOrWashing::965h5"|>, "Bathtub (filling or washing)" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Domestic sounds, home sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0006720204806241714, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.201141817091717 - 10.372816538362644*x))^(-1)], 
     "AudioSetID" -> "/m/03dnzn", "EntityCanonicalName" -> 
      "BathtubFillingOrWashing::33p3z"|>, "Hair dryer" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Domestic sounds, home sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00013308342095505132, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.266477761096798 - 33.24008984486396*x))^(-1)], 
     "AudioSetID" -> "/m/03wvsk", "EntityCanonicalName" -> 
      "HairDryer::r9dc6"|>, "Toilet flush" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Domestic sounds, home sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.000996601801197751, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.926094300561283 - 12.023087718792205*x))^(-1)], 
     "AudioSetID" -> "/m/01jt3m", "EntityCanonicalName" -> 
      "ToiletFlush::b5f2x"|>, "Toothbrush" -> 
    <|"DepthIndex" -> 0.6666666666666666, "FirstChildren" -> 
      {"Electric toothbrush"}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> 
      {"Electric toothbrush"}, "FlattenedParents" -> 
      {"Domestic sounds, home sounds", "Sounds of things"}, 
     "BottomDepth" -> 1, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 3.5556639186464097*^-6, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(12.08113933871148 - 30899.895919125283*x))^(-1)], 
     "AudioSetID" -> "/m/012xff", "EntityCanonicalName" -> 
      "Toothbrush::tgz55"|>, "Electric toothbrush" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Toothbrush"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Toothbrush", "Domestic sounds, home sounds", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00008635183802426995, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.703634547419593 - 66.41760825868228*x))^(-1)], 
     "AudioSetID" -> "/m/04fgwm", "EntityCanonicalName" -> 
      "ElectricToothbrush::kvc69"|>, "Vacuum cleaner" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Domestic sounds, home sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0009127897231153712, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.787606350467589 - 12.219520956887626*x))^(-1)], 
     "AudioSetID" -> "/m/0d31p", "EntityCanonicalName" -> 
      "VacuumCleaner::bqn86"|>, "Zipper (clothing)" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Domestic sounds, home sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00012140052522235599, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.539361438648356 - 370.7472509096079*x))^(-1)], 
     "AudioSetID" -> "/m/01s0vc", "EntityCanonicalName" -> 
      "ZipperClothing::t9rrq"|>, "Velcro, hook and loop fastener" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Domestic sounds, home sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/0zmy2j9", "EntityCanonicalName" -> 
      "VelcroHookAndLoopFastener::4445v"|>, "Keys jangling" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Domestic sounds, home sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00019048199564177193, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.638953060329564 - 100.9735264244616*x))^(-1)], 
     "AudioSetID" -> "/m/03v3yw", "EntityCanonicalName" -> 
      "KeysJangling::5frjp"|>, "Coin (dropping)" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Domestic sounds, home sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00013511522890856357, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.204045306894647 - 157.14883748458803*x))^(-1)], 
     "AudioSetID" -> "/m/0242l", "EntityCanonicalName" -> 
      "CoinDropping::s3q2k"|>, "Packing tape, duct tape" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Domestic sounds, home sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/05mxj0q", "EntityCanonicalName" -> 
      "PackingTapeDuctTape::g5r82"|>, "Scissors" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Domestic sounds, home sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00009244726188480665, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.382092265246104 - 216.33037517225708*x))^(-1)], 
     "AudioSetID" -> "/m/01lsmm", "EntityCanonicalName" -> 
      "Scissors::3792s"|>, "Electric shaver, electric razor" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Domestic sounds, home sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00036166181572517766, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.843272152156864 - 13.569580368627443*x))^(-1)], 
     "AudioSetID" -> "/m/02g901", "EntityCanonicalName" -> 
      "ElectricShaverElectricRazor::6s96q"|>, "Shuffling cards" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Domestic sounds, home sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00009193930989642859, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.962915746072518 - 562.7358023523445*x))^(-1)], 
     "AudioSetID" -> "/m/05rj2", "EntityCanonicalName" -> 
      "ShufflingCards::2v55k"|>, "Typing" -> 
    <|"DepthIndex" -> 0.6666666666666666, "FirstChildren" -> 
      {"Typewriter", "Computer keyboard"}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> 
      {"Typewriter", "Computer keyboard"}, "FlattenedParents" -> 
      {"Domestic sounds, home sounds", "Sounds of things"}, 
     "BottomDepth" -> 1, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0009625690179764208, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.51762896081899 - 9.518112623887731*x))^(-1)], 
     "AudioSetID" -> "/m/0316dw", "EntityCanonicalName" -> "Typing::895nm"|>, 
   "Typewriter" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Typing"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Typing", "Domestic sounds, home sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0002504203302703828, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.975094240630153 - 13.57345100370147*x))^(-1)], 
     "AudioSetID" -> "/m/0c2wf", "EntityCanonicalName" -> 
      "Typewriter::9m5j7"|>, "Computer keyboard" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Typing"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Typing", "Domestic sounds, home sounds", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0009265044268015788, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.03245966411543 - 9.636920717347415*x))^(-1)], 
     "AudioSetID" -> "/m/01m2v", "EntityCanonicalName" -> 
      "ComputerKeyboard::4hssh"|>, "Writing" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> 
      {"Domestic sounds, home sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Domestic sounds, home sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0003550584398762629, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.198264534060042 - 39.525857961141355*x))^(-1)], 
     "AudioSetID" -> "/m/081rb", "EntityCanonicalName" -> "Writing::xr57w"|>, 
   "Alarm" -> <|"DepthIndex" -> 0.3333333333333333, 
     "FirstChildren" -> {"Telephone", "Alarm clock", "Siren", "Doorbell", 
       "Buzzer", "Smoke detector, smoke alarm", "Fire alarm", "Car alarm", 
       "Vehicle horn, car horn, honking", "Bicycle bell", 
       "Air horn, truck horn", "Foghorn", "Whistle"}, 
     "FirstParents" -> {"Sounds of things"}, "FlattenedChildren" -> 
      {"Bicycle bell", "Vehicle horn, car horn, honking", "Car alarm", 
       "Air horn, truck horn", "Doorbell", "Telephone", "Alarm clock", 
       "Siren", "Buzzer", "Smoke detector, smoke alarm", "Fire alarm", 
       "Foghorn", "Whistle", "Toot", "Ding-dong", "Telephone bell ringing", 
       "Ringtone", "Cellphone buzz, vibrating alert", 
       "Telephone dialing, DTMF", "Dial tone", "Busy signal", 
       "Police car (siren)", "Ambulance (siren)", 
       "Fire engine, fire truck (siren)", "Civil defense siren", 
       "Kettle whistle", "Steam whistle"}, "FlattenedParents" -> 
      {"Sounds of things"}, "BottomDepth" -> 2, "TopDepth" -> 1, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00032712108051546966, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.571584490791869 - 19.98430866503204*x))^(-1)], 
     "AudioSetID" -> "/m/07pp_mv", "EntityCanonicalName" -> "Alarm::g33h3"|>, 
   "Telephone" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Telephone bell ringing", "Ringtone", 
       "Cellphone buzz, vibrating alert", "Telephone dialing, DTMF", 
       "Dial tone", "Busy signal"}, "FirstParents" -> {"Alarm"}, 
     "FlattenedChildren" -> {"Telephone bell ringing", "Ringtone", 
       "Cellphone buzz, vibrating alert", "Telephone dialing, DTMF", 
       "Dial tone", "Busy signal"}, "FlattenedParents" -> 
      {"Alarm", "Sounds of things"}, "BottomDepth" -> 1, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0005678903230066694, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.8959879176353 - 11.278596170701869*x))^(-1)], 
     "AudioSetID" -> "/m/07cx4", "EntityCanonicalName" -> 
      "Telephone::hm7k9"|>, "Telephone bell ringing" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Telephone"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Telephone", "Alarm", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0002814054015614444, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.978878657199834 - 12.748229937170441*x))^(-1)], 
     "AudioSetID" -> "/m/07pp8cl", "EntityCanonicalName" -> 
      "TelephoneBellRinging::mmvx8"|>, "Ringtone" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Telephone"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Telephone", "Alarm", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0004632522134007894, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.981398958586295 - 10.312638005964935*x))^(-1)], 
     "AudioSetID" -> "/m/01hnzm", "EntityCanonicalName" -> 
      "Ringtone::4zdms"|>, "Cellphone buzz, vibrating alert" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Telephone"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Telephone", "Alarm", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/01sb50", "EntityCanonicalName" -> 
      "CellphoneBuzzVibratingAlert::p2tzr"|>, "Telephone dialing, DTMF" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Telephone"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Telephone", "Alarm", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.00015949692435071036, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(9.359625506018219 - 14.31184551827856*x))^(-1)], 
     "AudioSetID" -> "/m/02c8p", "EntityCanonicalName" -> 
      "TelephoneDialingDTMF::239y2"|>, "Dial tone" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Telephone"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Telephone", "Alarm", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.00012394028516424628, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(9.398203646749627 - 17.668466865056036*x))^(-1)], 
     "AudioSetID" -> "/m/015jpf", "EntityCanonicalName" -> 
      "DialTone::y8qdr"|>, "Busy signal" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Telephone"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Telephone", "Alarm", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.00007619279825670877, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(9.843046960280892 - 71.69167080229292*x))^(-1)], 
     "AudioSetID" -> "/m/01z47d", "EntityCanonicalName" -> 
      "BusySignal::ttn4w"|>, "Alarm clock" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Alarm"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Alarm", "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00022553068283985797, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.93272704622093 - 16.142469623949005*x))^(-1)], 
     "AudioSetID" -> "/m/046dlr", "EntityCanonicalName" -> 
      "AlarmClock::nd484"|>, "Siren" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Police car (siren)", "Ambulance (siren)", 
       "Fire engine, fire truck (siren)", "Civil defense siren"}, 
     "FirstParents" -> {"Alarm"}, "FlattenedChildren" -> 
      {"Police car (siren)", "Ambulance (siren)", 
       "Fire engine, fire truck (siren)", "Civil defense siren"}, 
     "FlattenedParents" -> {"Alarm", "Sounds of things"}, "BottomDepth" -> 1, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.004011804804209906, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.500576236271877 - 9.97214107247734*x))^(-1)], 
     "AudioSetID" -> "/m/03kmc9", "EntityCanonicalName" -> "Siren::kmb4d"|>, 
   "Civil defense siren" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Siren"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Siren", "Alarm", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0009239646668596884, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.744742723412463 - 13.41292235498178*x))^(-1)], 
     "AudioSetID" -> "/m/0dgbq", "EntityCanonicalName" -> 
      "CivilDefenseSiren::m24x2"|>, "Buzzer" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Alarm"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Alarm", "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0004210921983654105, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.979973599213047 - 16.90438111697123*x))^(-1)], 
     "AudioSetID" -> "/m/030rvx", "EntityCanonicalName" -> "Buzzer::3psvy"|>, 
   "Smoke detector, smoke alarm" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Alarm"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Alarm", "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00020978417120013815, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.166703964027327 - 19.95531815245134*x))^(-1)], 
     "AudioSetID" -> "/m/01y3hg", "EntityCanonicalName" -> 
      "SmokeDetectorSmokeAlarm::8wsgq"|>, "Fire alarm" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> {"Alarm"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Alarm", "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00039975821485353205, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.622876308416647 - 11.817067286365887*x))^(-1)], 
     "AudioSetID" -> "/m/0c3f7m", "EntityCanonicalName" -> 
      "FireAlarm::458z9"|>, "Foghorn" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Alarm"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Alarm", "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0000477474869075375, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(10.412111377336728 - 119.70206920258367*x))^(-1)], 
     "AudioSetID" -> "/m/04fq5q", "EntityCanonicalName" -> 
      "Foghorn::g4z7x"|>, "Whistle" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Kettle whistle", "Steam whistle"}, 
     "FirstParents" -> {"Alarm"}, "FlattenedChildren" -> 
      {"Kettle whistle", "Steam whistle"}, "FlattenedParents" -> 
      {"Alarm", "Sounds of things"}, "BottomDepth" -> 1, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0002621032260030782, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.536988393207599 - 18.84727663286246*x))^(-1)], 
     "AudioSetID" -> "/m/0l156k", "EntityCanonicalName" -> 
      "Whistle::dbts3"|>, "Steam whistle" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Whistle"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Whistle", "Alarm", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.00034439144812032366, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.924020711740207 - 13.591778850724474*x))^(-1)], 
     "AudioSetID" -> "/m/06hck5", "EntityCanonicalName" -> 
      "SteamWhistle::t2883"|>, "Mechanisms" -> 
    <|"DepthIndex" -> 0.3333333333333333, "FirstChildren" -> 
      {"Ratchet, pawl", "Clock", "Gears", "Pulleys", "Sewing machine", 
       "Mechanical fan", "Air conditioning", "Cash register", "Printer", 
       "Camera"}, "FirstParents" -> {"Sounds of things"}, 
     "FlattenedChildren" -> {"Ratchet, pawl", "Clock", "Gears", "Pulleys", 
       "Sewing machine", "Mechanical fan", "Air conditioning", 
       "Cash register", "Printer", "Camera", "Tick", "Tick-tock", 
       "Single-lens reflex camera"}, "FlattenedParents" -> 
      {"Sounds of things"}, "BottomDepth" -> 2, "TopDepth" -> 1, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0007786903981835637, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.402695111961359 - 35.25678537265445*x))^(-1)], 
     "AudioSetID" -> "/t/dd00077", "EntityCanonicalName" -> 
      "Mechanisms::3cxy9"|>, "Ratchet, pawl" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Mechanisms"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Mechanisms", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0002900405853638714, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.541594558502245 - 28.80594571709975*x))^(-1)], 
     "AudioSetID" -> "/m/02bm9n", "EntityCanonicalName" -> 
      "RatchetPawl::q7wwj"|>, "Clock" -> 
    <|"DepthIndex" -> 0.6666666666666666, "FirstChildren" -> 
      {"Tick", "Tick-tock"}, "FirstParents" -> {"Mechanisms"}, 
     "FlattenedChildren" -> {"Tick", "Tick-tock"}, "FlattenedParents" -> 
      {"Mechanisms", "Sounds of things"}, "BottomDepth" -> 1, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.00030680300098034734, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.69478332196954 - 16.787280290340526*x))^(-1)], 
     "AudioSetID" -> "/m/01x3z", "EntityCanonicalName" -> "Clock::5hrsm"|>, 
   "Tick" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Clock", "Clicking"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Clock", "Clicking", "Mechanisms", 
       "Onomatopoeia", "Sounds of things", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> <|"Clock" -> 3, "Clicking" -> 3|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0008086595654978691, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.580096620773811 - 13.348287597558183*x))^(-1)], 
     "AudioSetID" -> "/m/07qjznt", "EntityCanonicalName" -> "Tick::hth53"|>, 
   "Tick-tock" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Clock"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Clock", "Mechanisms", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0013846771203185875, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.385717895688616 - 9.397195185020855*x))^(-1)], 
     "AudioSetID" -> "/m/07qjznl", "EntityCanonicalName" -> 
      "Tick-tock::3tp92"|>, "Gears" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Mechanisms"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Mechanisms", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0002458487623749803, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.604526068604153 - 33.32158649330071*x))^(-1)], 
     "AudioSetID" -> "/m/0l7xg", "EntityCanonicalName" -> "Gears::76w29"|>, 
   "Pulleys" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Mechanisms"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Mechanisms", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.000015238559651341754, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(12.07051288984144 - 11545.928822963086*x))^(-1)], 
     "AudioSetID" -> "/m/05zc1", "EntityCanonicalName" -> "Pulleys::j5xgm"|>, 
   "Sewing machine" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Mechanisms"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Mechanisms", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0008691058521148581, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.496544106113429 - 15.577112976477851*x))^(-1)], 
     "AudioSetID" -> "/m/0llzx", "EntityCanonicalName" -> 
      "SewingMachine::6z688"|>, "Mechanical fan" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Mechanisms"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Mechanisms", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0003550584398762629, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.140132725570401 - 109.79653365497745*x))^(-1)], 
     "AudioSetID" -> "/m/02x984l", "EntityCanonicalName" -> 
      "MechanicalFan::8t53b"|>, "Air conditioning" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Mechanisms"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Mechanisms", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0001407027007807222, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.982316680186193 - 132.53458573921137*x))^(-1)], 
     "AudioSetID" -> "/m/025wky1", "EntityCanonicalName" -> 
      "AirConditioning::5f858"|>, "Cash register" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Mechanisms"}, "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Mechanisms", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.00012343233317586822, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(9.538566283317248 - 21.2970802775708*x))^(-1)], 
     "AudioSetID" -> "/m/024dl", "EntityCanonicalName" -> 
      "CashRegister::648hw"|>, "Printer" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Mechanisms"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Mechanisms", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0016284940747400556, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.242863835209632 - 9.385206375142953*x))^(-1)], 
     "AudioSetID" -> "/m/01m4t", "EntityCanonicalName" -> "Printer::s52jh"|>, 
   "Camera" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Single-lens reflex camera"}, 
     "FirstParents" -> {"Mechanisms"}, "FlattenedChildren" -> 
      {"Single-lens reflex camera"}, "FlattenedParents" -> 
      {"Mechanisms", "Sounds of things"}, "BottomDepth" -> 1, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.00015035378855990532, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(9.202502725232417 - 190.10342625704476*x))^(-1)], 
     "AudioSetID" -> "/m/0dv5r", "EntityCanonicalName" -> "Camera::hpj2p"|>, 
   "Single-lens reflex camera" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Camera"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Camera", "Mechanisms", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.00017422753201367408, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(9.053778798159449 - 291.0213202276108*x))^(-1)], 
     "AudioSetID" -> "/m/07bjf", "EntityCanonicalName" -> 
      "Single-lensReflexCamera::s39n9"|>, 
   "Tools" -> <|"DepthIndex" -> 0.25, "FirstChildren" -> 
      {"Hammer", "Jackhammer", "Sawing", "Filing (rasp)", "Sanding", 
       "Power tool"}, "FirstParents" -> {"Sounds of things"}, 
     "FlattenedChildren" -> {"Hammer", "Jackhammer", "Sawing", 
       "Filing (rasp)", "Sanding", "Power tool", "Drill", 
       "Dental drill, dentist's drill"}, "FlattenedParents" -> 
      {"Sounds of things"}, "BottomDepth" -> 3, "TopDepth" -> 1, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0039036110306853796, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.408391192903394 - 11.149950810017561*x))^(-1)], 
     "AudioSetID" -> "/m/07k1x", "EntityCanonicalName" -> "Tools::863m9"|>, 
   "Hammer" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Tools"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Tools", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.00019149789961852805, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.875524040068038 - 30.804523621295363*x))^(-1)], 
     "AudioSetID" -> "/m/03l9g", "EntityCanonicalName" -> "Hammer::k88sd"|>, 
   "Jackhammer" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Tools"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Tools", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.00008025641416373324, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(10.621603824586572 - 130.57207177310067*x))^(-1)], 
     "AudioSetID" -> "/m/03p19w", "EntityCanonicalName" -> 
      "Jackhammer::7449y"|>, "Sawing" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Tools"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Tools", "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00034134373619005534, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.425092277323724 - 24.327195104207295*x))^(-1)], 
     "AudioSetID" -> "/m/01b82r", "EntityCanonicalName" -> "Sawing::j7834"|>, 
   "Filing (rasp)" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Tools"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Tools", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.00028902468138711527, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.831658058735227 - 12.708778315786805*x))^(-1)], 
     "AudioSetID" -> "/m/02p01q", "EntityCanonicalName" -> 
      "FilingRasp::c6tr8"|>, "Sanding" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Tools"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Tools", "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.000035556639186464096, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(10.649117271545565 - 40.20869558374751*x))^(-1)], 
     "AudioSetID" -> "/m/023vsd", "EntityCanonicalName" -> 
      "Sanding::qc4bx"|>, "Power tool" -> <|"DepthIndex" -> 0.5, 
     "FirstChildren" -> {"Drill"}, "FirstParents" -> {"Tools"}, 
     "FlattenedChildren" -> {"Drill", "Dental drill, dentist's drill"}, 
     "FlattenedParents" -> {"Tools", "Sounds of things"}, "BottomDepth" -> 2, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0015502694685298345, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.437829772483563 - 16.2529903137637*x))^(-1)], 
     "AudioSetID" -> "/m/0_ksk", "EntityCanonicalName" -> 
      "PowerTool::w25gt"|>, "Drill" -> <|"DepthIndex" -> 0.75, 
     "FirstChildren" -> {"Dental drill, dentist's drill"}, 
     "FirstParents" -> {"Power tool"}, "FlattenedChildren" -> 
      {"Dental drill, dentist's drill"}, "FlattenedParents" -> 
      {"Power tool", "Tools", "Sounds of things"}, "BottomDepth" -> 1, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0009193930989642859, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.659425040964368 - 16.637532420578722*x))^(-1)], 
     "AudioSetID" -> "/m/01d380", "EntityCanonicalName" -> "Drill::43p23"|>, 
   "Explosion" -> <|"DepthIndex" -> 0.3333333333333333, 
     "FirstChildren" -> {"Gunshot, gunfire", "Fireworks", "Burst, pop", 
       "Eruption", "Boom"}, "FirstParents" -> {"Sounds of things"}, 
     "FlattenedChildren" -> {"Gunshot, gunfire", "Fireworks", "Burst, pop", 
       "Eruption", "Boom", "Machine gun", "Fusillade", "Artillery fire", 
       "Cap gun", "Firecracker", "Sonic boom"}, "FlattenedParents" -> 
      {"Sounds of things"}, "BottomDepth" -> 2, "TopDepth" -> 1, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0010519685679309591, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.374793344409591 - 22.054763800480337*x))^(-1)], 
     "AudioSetID" -> "/m/014zdl", "EntityCanonicalName" -> 
      "Explosion::5xbzd"|>, "Gunshot, gunfire" -> 
    <|"DepthIndex" -> 0.6666666666666666, "FirstChildren" -> 
      {"Machine gun", "Fusillade", "Artillery fire", "Cap gun"}, 
     "FirstParents" -> {"Explosion"}, "FlattenedChildren" -> 
      {"Machine gun", "Fusillade", "Artillery fire", "Cap gun"}, 
     "FlattenedParents" -> {"Explosion", "Sounds of things"}, 
     "BottomDepth" -> 1, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.001885009828870975, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.083164998015935 - 14.60895780628041*x))^(-1)], 
     "AudioSetID" -> "/m/032s66", "EntityCanonicalName" -> 
      "GunshotGunfire::2t85r"|>, "Machine gun" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Gunshot, gunfire"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Gunshot, gunfire", "Explosion", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {}, "ClassPrior" -> 0.000842184396730821, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.841128390779634 - 16.40426712122435*x))^(-1)], 
     "AudioSetID" -> "/m/04zjc", "EntityCanonicalName" -> 
      "MachineGun::t6sx4"|>, "Fusillade" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Gunshot, gunfire"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Gunshot, gunfire", "Explosion", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> 0.0007527848467762827, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.060403143167276 - 22.00015187644212*x))^(-1)], 
     "AudioSetID" -> "/m/02z32qm", "EntityCanonicalName" -> 
      "Fusillade::8739p"|>, "Artillery fire" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Gunshot, gunfire"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Gunshot, gunfire", "Explosion", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00040839339865595905, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.4206954529868 - 37.19729874088106*x))^(-1)], 
     "AudioSetID" -> "/m/0_1c", "EntityCanonicalName" -> 
      "ArtilleryFire::ryf28"|>, "Cap gun" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Gunshot, gunfire"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Gunshot, gunfire", "Explosion", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00026718274588685876, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.967984463234602 - 15.364643424618036*x))^(-1)], 
     "AudioSetID" -> "/m/073cg4", "EntityCanonicalName" -> "CapGun::789h9"|>, 
   "Fireworks" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Firecracker"}, "FirstParents" -> {"Explosion"}, 
     "FlattenedChildren" -> {"Firecracker"}, "FlattenedParents" -> 
      {"Explosion", "Sounds of things"}, "BottomDepth" -> 1, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.001441059791028552, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.007340401726335 - 11.26394968676404*x))^(-1)], 
     "AudioSetID" -> "/m/0g6b5", "EntityCanonicalName" -> 
      "Fireworks::4hsvt"|>, "Firecracker" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Fireworks"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Fireworks", "Explosion", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0005734777948788281, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.975640116251908 - 15.307332567341684*x))^(-1)], 
     "AudioSetID" -> "/g/122z_qxw", "EntityCanonicalName" -> 
      "Firecracker::8zk23"|>, "Burst, pop" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Explosion"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Explosion", "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.001050952663954203, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.228809049832312 - 73.10565527827826*x))^(-1)], 
     "AudioSetID" -> "/m/07qsvvw", "EntityCanonicalName" -> 
      "BurstPop::7n482"|>, "Eruption" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Explosion"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Explosion", "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0003332165043760064, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.364769717656017 - 43.51906487855507*x))^(-1)], 
     "AudioSetID" -> "/m/07pxg6y", "EntityCanonicalName" -> 
      "Eruption::kcmv3"|>, "Boom" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Sonic boom"}, "FirstParents" -> {"Explosion"}, 
     "FlattenedChildren" -> {"Sonic boom"}, "FlattenedParents" -> 
      {"Explosion", "Sounds of things"}, "BottomDepth" -> 1, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0007532927987646608, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.16877323928263 - 28.945308762643318*x))^(-1)], 
     "AudioSetID" -> "/m/07qqyl4", "EntityCanonicalName" -> "Boom::89k4k"|>, 
   "Sonic boom" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Boom"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Boom", "Explosion", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/0193bn", "EntityCanonicalName" -> 
      "SonicBoom::k2bq2"|>, "Wood" -> <|"DepthIndex" -> 0.5, 
     "FirstChildren" -> {"Chop", "Splinter", "Crack", "Snap"}, 
     "FirstParents" -> {"Sounds of things"}, "FlattenedChildren" -> 
      {"Chop", "Splinter", "Crack", "Snap"}, "FlattenedParents" -> 
      {"Sounds of things"}, "BottomDepth" -> 1, "TopDepth" -> 1, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0015558569404019932, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.99258573953042 - 17.499929989617023*x))^(-1)], 
     "AudioSetID" -> "/m/083vt", "EntityCanonicalName" -> "Wood::x6x26"|>, 
   "Chop" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Wood"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Wood", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.00015848102037395426, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.904057179415037 - 173.66394382554844*x))^(-1)], 
     "AudioSetID" -> "/m/07pczhz", "EntityCanonicalName" -> "Chop::q5nf9"|>, 
   "Splinter" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Wood"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Wood", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.000016254463628097872, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(11.8192588510279 + 1336.7843846957446*x))^(-1)], 
     "AudioSetID" -> "/m/07pl1bw", "EntityCanonicalName" -> 
      "Splinter::y8647"|>, "Crack" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Wood", "Onomatopoeia"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Wood", "Onomatopoeia", "Sounds of things", 
       "Source-ambiguous sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> <|"Wood" -> 2, "Onomatopoeia" -> 2|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00006908147041941596, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(10.471608991852332 - 302.0176523737059*x))^(-1)], 
     "AudioSetID" -> "/m/07qs1cx", "EntityCanonicalName" -> "Crack::936rp"|>, 
   "Snap" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Wood", "Onomatopoeia"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Wood", "Onomatopoeia", "Sounds of things", 
       "Source-ambiguous sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> <|"Wood" -> 2, "Onomatopoeia" -> 2|>, 
     "Restrictions" -> {}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/07pc8l3", "EntityCanonicalName" -> "Snap::8695g"|>, 
   "Glass" -> <|"DepthIndex" -> 0.5, "FirstChildren" -> 
      {"Chink, clink", "Shatter"}, "FirstParents" -> {"Sounds of things"}, 
     "FlattenedChildren" -> {"Chink, clink", "Shatter"}, 
     "FlattenedParents" -> {"Sounds of things"}, "BottomDepth" -> 1, 
     "TopDepth" -> 1, "Restrictions" -> {}, "ClassPrior" -> 
      0.00035759819981815317, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.9989253711314925 - 48.72316365526825*x))^(-1)], 
     "AudioSetID" -> "/m/039jq", "EntityCanonicalName" -> "Glass::cw979"|>, 
   "Chink, clink" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Glass"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Glass", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.00032254951262006715, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.308399144710462 - 44.54978102331615*x))^(-1)], 
     "AudioSetID" -> "/m/07q7njn", "EntityCanonicalName" -> 
      "ChinkClink::j4453"|>, "Shatter" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Glass"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Glass", "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00011530510136181928, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.309934374792242 - 325.431180996605*x))^(-1)], 
     "AudioSetID" -> "/m/07rn7sz", "EntityCanonicalName" -> 
      "Shatter::ccnv5"|>, "Liquid" -> <|"DepthIndex" -> 0.3333333333333333, 
     "FirstChildren" -> {"Splash, splatter", "Squish", "Drip", "Pour", 
       "Fill (with liquid)", "Spray", "Pump (liquid)", "Stir", "Boiling"}, 
     "FirstParents" -> {"Sounds of things"}, "FlattenedChildren" -> 
      {"Splash, splatter", "Squish", "Drip", "Pour", "Fill (with liquid)", 
       "Spray", "Pump (liquid)", "Stir", "Boiling", "Slosh", 
       "Trickle, dribble", "Gush"}, "FlattenedParents" -> 
      {"Sounds of things"}, "BottomDepth" -> 2, "TopDepth" -> 1, 
     "Restrictions" -> {}, "ClassPrior" -> 0.000772594974323027, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.707043285973903 - 21.304525806709876*x))^(-1)], 
     "AudioSetID" -> "/m/04k94", "EntityCanonicalName" -> "Liquid::f3824"|>, 
   "Splash, splatter" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Slosh"}, "FirstParents" -> {"Liquid"}, 
     "FlattenedChildren" -> {"Slosh"}, "FlattenedParents" -> 
      {"Liquid", "Sounds of things"}, "BottomDepth" -> 1, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0004002661668419101, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.391502973930235 - 17.89042732589296*x))^(-1)], 
     "AudioSetID" -> "/m/07rrlb6", "EntityCanonicalName" -> 
      "SplashSplatter::pfc4d"|>, "Slosh" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Splash, splatter"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Splash, splatter", "Liquid", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0005424927235877665, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.913143928528528 - 16.38276258638841*x))^(-1)], 
     "AudioSetID" -> "/m/07p6mqd", "EntityCanonicalName" -> "Slosh::k9qn3"|>, 
   "Squish" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Liquid", "Onomatopoeia"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Liquid", "Onomatopoeia", "Sounds of things", 
       "Source-ambiguous sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> <|"Liquid" -> 2, "Onomatopoeia" -> 2|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0001259720931177585, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.88943905683479 - 463.91788837092435*x))^(-1)], 
     "AudioSetID" -> "/m/07qlwh6", "EntityCanonicalName" -> 
      "Squish::6m9r5"|>, "Drip" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Liquid"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Liquid", "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0003195018006897988, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.507390702630383 - 19.508640262752564*x))^(-1)], 
     "AudioSetID" -> "/m/07r5v4s", "EntityCanonicalName" -> "Drip::5r979"|>, 
   "Pour" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Trickle, dribble", "Gush"}, 
     "FirstParents" -> {"Liquid"}, "FlattenedChildren" -> 
      {"Trickle, dribble", "Gush"}, "FlattenedParents" -> 
      {"Liquid", "Sounds of things"}, "BottomDepth" -> 1, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0001361311328853197, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.426039214037502 - 34.70318056631488*x))^(-1)], 
     "AudioSetID" -> "/m/07prgkl", "EntityCanonicalName" -> "Pour::9n62q"|>, 
   "Trickle, dribble" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Pour"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Pour", "Liquid", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.000598875394297731, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.845254362755285 - 24.040664696722015*x))^(-1)], 
     "AudioSetID" -> "/m/07pqc89", "EntityCanonicalName" -> 
      "TrickleDribble::39b23"|>, "Gush" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Pour"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Pour", "Liquid", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.00014375041271099057, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(9.01376184386064 - 79.07183186743953*x))^(-1)], 
     "AudioSetID" -> "/t/dd00088", "EntityCanonicalName" -> "Gush::pbc59"|>, 
   "Fill (with liquid)" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Liquid"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Liquid", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00024991237828200476, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.643056669601481 - 43.64186820212993*x))^(-1)], 
     "AudioSetID" -> "/m/07p7b8y", "EntityCanonicalName" -> 
      "FillWithLiquid::n34n6"|>, "Spray" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Liquid"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Liquid", "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0014618858225520523, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(6.923183523274285 - 15.997400606734262*x))^(-1)], 
     "AudioSetID" -> "/m/07qlf79", "EntityCanonicalName" -> "Spray::353m2"|>, 
   "Pump (liquid)" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Liquid"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Liquid", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00024026129050282166, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.590932755095963 - 32.395476338813125*x))^(-1)], 
     "AudioSetID" -> "/m/07ptzwd", "EntityCanonicalName" -> 
      "PumpLiquid::3bv43"|>, "Stir" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Liquid"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Liquid", "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0003438834961319456, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.75755625841626 - 11.607890735517161*x))^(-1)], 
     "AudioSetID" -> "/m/07ptfmf", "EntityCanonicalName" -> "Stir::9g75q"|>, 
   "Boiling" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Liquid"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Liquid", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00016305258826935677, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.026637746462205 - 121.4822120964839*x))^(-1)], 
     "AudioSetID" -> "/m/0dv3j", "EntityCanonicalName" -> "Boiling::y75j9"|>, 
   "Miscellaneous sources" -> <|"DepthIndex" -> 0.25, 
     "FirstChildren" -> {"Sonar", "Duck call (hunting tool)", "Arrow", 
       "Sound equipment"}, "FirstParents" -> {"Sounds of things"}, 
     "FlattenedChildren" -> {"Sonar", "Duck call (hunting tool)", "Arrow", 
       "Sound equipment", "Whoosh, swoosh, swish", "Thump, thud", "Wobble", 
       "Microphone", "Electronic tuner", "Guitar amplifier", "Effects unit", 
       "Thunk", "Clunk", "Wind noise (microphone)", "Chorus effect"}, 
     "FlattenedParents" -> {"Sounds of things"}, "BottomDepth" -> 3, 
     "TopDepth" -> 1, "Restrictions" -> {"abstract"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/t/dd00089", "EntityCanonicalName" -> 
      "MiscellaneousSources::qb2r2"|>, 
   "Sonar" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Miscellaneous sources"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Miscellaneous sources", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00010565401358263617, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(10.051769638603794 - 52.38349421017425*x))^(-1)], 
     "AudioSetID" -> "/m/0790c", "EntityCanonicalName" -> "Sonar::77vk6"|>, 
   "Duck call (hunting tool)" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Miscellaneous sources"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Miscellaneous sources", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/04179zz", "EntityCanonicalName" -> 
      "DuckCallHuntingTool::qs8jj"|>, 
   "Arrow" -> <|"DepthIndex" -> 0.5, "FirstChildren" -> 
      {"Whoosh, swoosh, swish", "Thump, thud", "Wobble"}, 
     "FirstParents" -> {"Miscellaneous sources"}, "FlattenedChildren" -> 
      {"Whoosh, swoosh, swish", "Thump, thud", "Wobble", "Thunk", "Clunk"}, 
     "FlattenedParents" -> {"Miscellaneous sources", "Sounds of things"}, 
     "BottomDepth" -> 2, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00046934763726132607, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.895026771103 - 106.13490270271532*x))^(-1)], 
     "AudioSetID" -> "/m/0dl83", "EntityCanonicalName" -> "Arrow::5cj9x"|>, 
   "Whoosh, swoosh, swish" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Arrow", "Onomatopoeia"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Arrow", "Onomatopoeia", "Miscellaneous sources", 
       "Source-ambiguous sounds", "Sounds of things"}, "BottomDepth" -> 0, 
     "TopDepth" -> <|"Arrow" -> 3, "Onomatopoeia" -> 2|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0006308763695655487, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.51091293362102 - 37.77602443038754*x))^(-1)], 
     "AudioSetID" -> "/m/07rqsjt", "EntityCanonicalName" -> 
      "WhooshSwooshSwish::3g7fr"|>, "Thump, thud" -> 
    <|"DepthIndex" -> 0.7083333333333333, "FirstChildren" -> 
      {"Thunk", "Clunk"}, "FirstParents" -> 
      {"Arrow", "Generic impact sounds"}, "FlattenedChildren" -> 
      {"Thunk", "Clunk"}, "FlattenedParents" -> 
      {"Arrow", "Generic impact sounds", "Miscellaneous sources", 
       "Source-ambiguous sounds", "Sounds of things"}, "BottomDepth" -> 1, 
     "TopDepth" -> <|"Arrow" -> 3, "Generic impact sounds" -> 2|>, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0008264378850911012, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.393845438822529 - 13.857234788736559*x))^(-1)], 
     "AudioSetID" -> "/m/07qnq_y", "EntityCanonicalName" -> 
      "ThumpThud::h3zwb"|>, "Thunk" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Thump, thud"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Thump, thud", "Arrow", "Generic impact sounds", 
       "Miscellaneous sources", "Source-ambiguous sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 4, 
     "Restrictions" -> {}, "ClassPrior" -> 0.000147814028618015, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.940406030194486 - 155.92052299311794*x))^(-1)], 
     "AudioSetID" -> "/m/07rrh0c", "EntityCanonicalName" -> "Thunk::g3g5b"|>, 
   "Clunk" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Thump, thud"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Thump, thud", "Arrow", "Generic impact sounds", 
       "Miscellaneous sources", "Source-ambiguous sounds", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 4, 
     "Restrictions" -> {}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/t/dd00108", "EntityCanonicalName" -> "Clunk::gn8jc"|>, 
   "Wobble" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Arrow"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Arrow", "Miscellaneous sources", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/07pk7mg", "EntityCanonicalName" -> 
      "Wobble::fqmn4"|>, "Sound equipment" -> <|"DepthIndex" -> 0.5, 
     "FirstChildren" -> {"Microphone", "Electronic tuner", 
       "Guitar amplifier", "Effects unit"}, "FirstParents" -> 
      {"Miscellaneous sources"}, "FlattenedChildren" -> 
      {"Microphone", "Electronic tuner", "Guitar amplifier", "Effects unit", 
       "Wind noise (microphone)", "Chorus effect"}, 
     "FlattenedParents" -> {"Miscellaneous sources", "Sounds of things"}, 
     "BottomDepth" -> 2, "TopDepth" -> 2, "Restrictions" -> {"abstract"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/t/dd00091", "EntityCanonicalName" -> 
      "SoundEquipment::y9fwd"|>, "Microphone" -> <|"DepthIndex" -> 0.75, 
     "FirstChildren" -> {"Wind noise (microphone)"}, 
     "FirstParents" -> {"Sound equipment"}, "FlattenedChildren" -> 
      {"Wind noise (microphone)"}, "FlattenedParents" -> 
      {"Sound equipment", "Miscellaneous sources", "Sounds of things"}, 
     "BottomDepth" -> 1, "TopDepth" -> 3, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/0hg7b", "EntityCanonicalName" -> 
      "Microphone::h5fx8"|>, "Electronic tuner" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Sound equipment"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Sound equipment", "Miscellaneous sources", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {"blacklist"}, "ClassPrior" -> 0.0006577978249495858, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.9533260038541975 - 17.290446568694176*x))^(-1)], 
     "AudioSetID" -> "/m/0b_fwt", "EntityCanonicalName" -> 
      "ElectronicTuner::5652t"|>, "Guitar amplifier" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Sound equipment"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Sound equipment", "Miscellaneous sources", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {"blacklist"}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/01vfsf", "EntityCanonicalName" -> 
      "GuitarAmplifier::w5bb9"|>, "Effects unit" -> 
    <|"DepthIndex" -> 0.75, "FirstChildren" -> {"Chorus effect"}, 
     "FirstParents" -> {"Sound equipment"}, "FlattenedChildren" -> 
      {"Chorus effect"}, "FlattenedParents" -> {"Sound equipment", 
       "Miscellaneous sources", "Sounds of things"}, "BottomDepth" -> 1, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.002267497676119653, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.637919537852077 - 11.342272296474997*x))^(-1)], 
     "AudioSetID" -> "/m/02rr_", "EntityCanonicalName" -> 
      "EffectsUnit::9by3t"|>, "Chorus effect" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Effects unit"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Effects unit", "Sound equipment", "Miscellaneous sources", 
       "Sounds of things"}, "BottomDepth" -> 0, "TopDepth" -> 4, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00012038462124559986, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.435439505709544 - 72.17582214338707*x))^(-1)], 
     "AudioSetID" -> "/m/07m2kt", "EntityCanonicalName" -> 
      "ChorusEffect::m55ct"|>, "Specific impact sounds" -> 
    <|"DepthIndex" -> 0.5, "FirstChildren" -> {"Basketball bounce"}, 
     "FirstParents" -> {"Sounds of things"}, "FlattenedChildren" -> 
      {"Basketball bounce"}, "FlattenedParents" -> {"Sounds of things"}, 
     "BottomDepth" -> 1, "TopDepth" -> 1, "Restrictions" -> {"abstract"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/t/dd00133", "EntityCanonicalName" -> 
      "SpecificImpactSounds::26x2h"|>, "Basketball bounce" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Specific impact sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Specific impact sounds", "Sounds of things"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0009569815461042623, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.910523819487363 - 10.967228832085102*x))^(-1)], 
     "AudioSetID" -> "/m/018w8", "EntityCanonicalName" -> 
      "BasketballBounce::h559f"|>, "Source-ambiguous sounds" -> 
    <|"DepthIndex" -> 0., "FirstChildren" -> {"Generic impact sounds", 
       "Surface contact", "Deformable shell", "Onomatopoeia", "Silence", 
       "Other sourceless"}, "FirstParents" -> {}, "FlattenedChildren" -> 
      {"Generic impact sounds", "Surface contact", "Deformable shell", 
       "Onomatopoeia", "Silence", "Other sourceless", "Knock", "Tap", 
       "Thump, thud", "Bang", "Slap, smack", "Whack, thwack", "Smash, crash", 
       "Breaking", "Bouncing", "Whip", "Flap", "Scratch", "Scrape", "Rub", 
       "Roll", "Grind", "Crushing", "Crumpling, crinkling", "Tearing", 
       "Hiss", "Rattle", "Crackle", "Crack", "Snap", "Squish", 
       "Whoosh, swoosh, swish", "Brief tone", "Creak", "Rustle", "Whir", 
       "Clatter", "Sizzle", "Clicking", "Rumble", "Blare", "Plop", 
       "Jingle, tinkle", "Fizz", "Puff", "Hum", "Zing", "Boing", "Crunch", 
       "Sine wave", "Sound effect", "Pulse", "Infrasound", 
       "Bass (frequency range)", "Ringing (of resonator)", "Thunk", "Clunk", 
       "Chirp, tweet", "Buzz", "Squeak", "Beep, bleep", "Ping", "Ding", 
       "Clang", "Twang", "Squeal", "Screech", "Clip-clop", "Tick", 
       "Clickety-clack", "Harmonic", "Chirp tone"}, "FlattenedParents" -> {}, 
     "BottomDepth" -> 3, "TopDepth" -> 0, "Restrictions" -> {"abstract"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/t/dd00098", "EntityCanonicalName" -> 
      "Source-ambiguousSounds::ck9g9"|>, "Generic impact sounds" -> 
    <|"DepthIndex" -> 0.3333333333333333, "FirstChildren" -> 
      {"Bang", "Slap, smack", "Whack, thwack", "Smash, crash", "Breaking", 
       "Bouncing", "Knock", "Tap", "Thump, thud", "Whip", "Flap"}, 
     "FirstParents" -> {"Source-ambiguous sounds"}, 
     "FlattenedChildren" -> {"Knock", "Tap", "Thump, thud", "Bang", 
       "Slap, smack", "Whack, thwack", "Smash, crash", "Breaking", 
       "Bouncing", "Whip", "Flap", "Thunk", "Clunk"}, 
     "FlattenedParents" -> {"Source-ambiguous sounds"}, "BottomDepth" -> 2, 
     "TopDepth" -> 1, "Restrictions" -> {"abstract"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/t/dd00099", "EntityCanonicalName" -> 
      "GenericImpactSounds::d2268"|>, 
   "Bang" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Generic impact sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Generic impact sounds", 
       "Source-ambiguous sounds"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00011632100533857539, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.330528797661575 - 380.9171314366164*x))^(-1)], 
     "AudioSetID" -> "/m/07pws3f", "EntityCanonicalName" -> "Bang::746g2"|>, 
   "Slap, smack" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Generic impact sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Generic impact sounds", 
       "Source-ambiguous sounds"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0003697890475392266, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.099302358682621 - 62.41885448406024*x))^(-1)], 
     "AudioSetID" -> "/m/07ryjzk", "EntityCanonicalName" -> 
      "SlapSmack::4745w"|>, "Whack, thwack" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Generic impact sounds"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Generic impact sounds", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0007009737439617207, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.390443951401455 - 14.610257301233819*x))^(-1)], 
     "AudioSetID" -> "/m/07rdhzs", "EntityCanonicalName" -> 
      "WhackThwack::sw9hq"|>, "Smash, crash" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Generic impact sounds"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Generic impact sounds", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0006694807206822811, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.727282048972955 - 10.66333835809785*x))^(-1)], 
     "AudioSetID" -> "/m/07pjjrj", "EntityCanonicalName" -> 
      "SmashCrash::9y9x9"|>, "Breaking" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Generic impact sounds"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Generic impact sounds", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0001701639161066496, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(9.253358432614991 - 177.58028002816036*x))^(-1)], 
     "AudioSetID" -> "/m/07pc8lb", "EntityCanonicalName" -> 
      "Breaking::5358k"|>, "Bouncing" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Generic impact sounds"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Generic impact sounds", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00004520772696564721, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(10.339678296760209 - 585.9517668726696*x))^(-1)], 
     "AudioSetID" -> "/m/07pqn27", "EntityCanonicalName" -> 
      "Bouncing::3g525"|>, "Whip" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Generic impact sounds"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Generic impact sounds", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00016305258826935677, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.310798703959332 - 52.340077285046696*x))^(-1)], 
     "AudioSetID" -> "/m/07rbp7_", "EntityCanonicalName" -> "Whip::3337r"|>, 
   "Flap" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Generic impact sounds"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Generic impact sounds", 
       "Source-ambiguous sounds"}, "BottomDepth" -> 0, "TopDepth" -> 2, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00017524343599043018, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.340300076641745 - 278.4307629124601*x))^(-1)], 
     "AudioSetID" -> "/m/07pyf11", "EntityCanonicalName" -> "Flap::5dmt3"|>, 
   "Surface contact" -> <|"DepthIndex" -> 0.5, "FirstChildren" -> 
      {"Scratch", "Scrape", "Rub", "Roll", "Grind"}, 
     "FirstParents" -> {"Source-ambiguous sounds"}, 
     "FlattenedChildren" -> {"Scratch", "Scrape", "Rub", "Roll", "Grind"}, 
     "FlattenedParents" -> {"Source-ambiguous sounds"}, "BottomDepth" -> 1, 
     "TopDepth" -> 1, "Restrictions" -> {"abstract"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/t/dd00109", "EntityCanonicalName" -> 
      "SurfaceContact::zpy23"|>, "Scratch" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Surface contact"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Surface contact", "Source-ambiguous sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0001346072769201855, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(9.394980313912969 - 37.821591738760354*x))^(-1)], 
     "AudioSetID" -> "/m/07qb_dv", "EntityCanonicalName" -> 
      "Scratch::7b638"|>, "Scrape" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Surface contact"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Surface contact", "Source-ambiguous sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.00014730607662963698, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(9.65529627659049 - 502.4165103606329*x))^(-1)], 
     "AudioSetID" -> "/m/07qv4k0", "EntityCanonicalName" -> 
      "Scrape::69m29"|>, "Rub" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Surface contact"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Surface contact", "Source-ambiguous sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0006882749442522693, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.726258556695884 - 11.115002389835274*x))^(-1)], 
     "AudioSetID" -> "/m/07pdjhy", "EntityCanonicalName" -> "Rub::ymzm8"|>, 
   "Roll" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Surface contact"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Surface contact", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> 0.0009554576901391281, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.44951877140896 - 58.06243772276393*x))^(-1)], 
     "AudioSetID" -> "/m/07s8j8t", "EntityCanonicalName" -> "Roll::h8hkj"|>, 
   "Grind" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Surface contact"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Surface contact", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/07pt6mm", "EntityCanonicalName" -> "Grind::qs8st"|>, 
   "Deformable shell" -> <|"DepthIndex" -> 0.5, "FirstChildren" -> 
      {"Crushing", "Crumpling, crinkling", "Tearing"}, 
     "FirstParents" -> {"Source-ambiguous sounds"}, 
     "FlattenedChildren" -> {"Crushing", "Crumpling, crinkling", "Tearing"}, 
     "FlattenedParents" -> {"Source-ambiguous sounds"}, "BottomDepth" -> 1, 
     "TopDepth" -> 1, "Restrictions" -> {"abstract"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/t/dd00110", "EntityCanonicalName" -> 
      "DeformableShell::278h4"|>, "Crushing" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Deformable shell"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Deformable shell", "Source-ambiguous sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.000024889647430524867, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(10.494173738953249 - 233.97754235675424*x))^(-1)], 
     "AudioSetID" -> "/m/07plct2", "EntityCanonicalName" -> 
      "Crushing::426j5"|>, "Crumpling, crinkling" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Deformable shell"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Deformable shell", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0007329747192295384, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(9.095903752340298 - 10.850980260920448*x))^(-1)], 
     "AudioSetID" -> "/t/dd00112", "EntityCanonicalName" -> 
      "CrumplingCrinkling::z99qb"|>, "Tearing" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Deformable shell"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Deformable shell", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0001198766692572218, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(9.444564404567881 - 30.48916874871178*x))^(-1)], 
     "AudioSetID" -> "/m/07qcx4z", "EntityCanonicalName" -> 
      "Tearing::83r4s"|>, "Onomatopoeia" -> 
    <|"DepthIndex" -> 0.3333333333333333, "FirstChildren" -> 
      {"Brief tone", "Hiss", "Creak", "Rattle", "Whoosh, swoosh, swish", 
       "Rustle", "Whir", "Clatter", "Sizzle", "Clicking", "Rumble", "Blare", 
       "Plop", "Jingle, tinkle", "Fizz", "Puff", "Hum", "Squish", "Zing", 
       "Boing", "Crackle", "Crunch", "Crack", "Snap"}, 
     "FirstParents" -> {"Source-ambiguous sounds"}, 
     "FlattenedChildren" -> {"Hiss", "Rattle", "Crackle", "Crack", "Snap", 
       "Squish", "Whoosh, swoosh, swish", "Brief tone", "Creak", "Rustle", 
       "Whir", "Clatter", "Sizzle", "Clicking", "Rumble", "Blare", "Plop", 
       "Jingle, tinkle", "Fizz", "Puff", "Hum", "Zing", "Boing", "Crunch", 
       "Chirp, tweet", "Buzz", "Squeak", "Beep, bleep", "Ping", "Ding", 
       "Clang", "Twang", "Squeal", "Screech", "Clip-clop", "Tick", 
       "Clickety-clack"}, "FlattenedParents" -> {"Source-ambiguous sounds"}, 
     "BottomDepth" -> 2, "TopDepth" -> 1, "Restrictions" -> {"abstract"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/05n1m", "EntityCanonicalName" -> 
      "Onomatopoeia::ysj6k"|>, "Brief tone" -> 
    <|"DepthIndex" -> 0.6666666666666666, "FirstChildren" -> 
      {"Beep, bleep", "Ping", "Ding", "Clang", "Twang", "Chirp, tweet", 
       "Buzz", "Squeak", "Squeal", "Screech"}, "FirstParents" -> 
      {"Onomatopoeia"}, "FlattenedChildren" -> {"Chirp, tweet", "Buzz", 
       "Squeak", "Beep, bleep", "Ping", "Ding", "Clang", "Twang", "Squeal", 
       "Screech"}, "FlattenedParents" -> {"Onomatopoeia", 
       "Source-ambiguous sounds"}, "BottomDepth" -> 1, "TopDepth" -> 2, 
     "Restrictions" -> {"abstract"}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/t/dd000138", "EntityCanonicalName" -> 
      "BriefTone::2qxg6"|>, "Beep, bleep" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Brief tone"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Brief tone", "Onomatopoeia", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0007187520635549528, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.648750843010013 - 13.672384444814751*x))^(-1)], 
     "AudioSetID" -> "/m/02fs_r", "EntityCanonicalName" -> 
      "BeepBleep::7t475"|>, "Ping" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Brief tone"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Brief tone", "Onomatopoeia", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0003322006003992503, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.125758939683495 - 28.323260875026065*x))^(-1)], 
     "AudioSetID" -> "/m/07qwdck", "EntityCanonicalName" -> "Ping::nts36"|>, 
   "Ding" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Brief tone"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Brief tone", "Onomatopoeia", 
       "Source-ambiguous sounds"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {}, "ClassPrior" -> 0.0003062950489919693, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.774355418632478 - 19.30040775324285*x))^(-1)], 
     "AudioSetID" -> "/m/07phxs1", "EntityCanonicalName" -> "Ding::3vn99"|>, 
   "Clang" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Brief tone"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Brief tone", "Onomatopoeia", 
       "Source-ambiguous sounds"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00008889159796616024, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(10.39601585092764 - 63.801368498291275*x))^(-1)], 
     "AudioSetID" -> "/m/07rv4dm", "EntityCanonicalName" -> "Clang::39n7m"|>, 
   "Twang" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Brief tone"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Brief tone", "Onomatopoeia", 
       "Source-ambiguous sounds"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/07rpk1c", "EntityCanonicalName" -> "Twang::wsgn9"|>, 
   "Squeal" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Brief tone"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Brief tone", "Onomatopoeia", 
       "Source-ambiguous sounds"}, "BottomDepth" -> 0, "TopDepth" -> 3, 
     "Restrictions" -> {}, "ClassPrior" -> 0.00003047711930268351, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(11.110363918089648 - 1954.9755783924575*x))^(-1)], 
     "AudioSetID" -> "/m/07s02z0", "EntityCanonicalName" -> 
      "Squeal::652n5"|>, "Screech" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Brief tone"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Brief tone", "Onomatopoeia", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/07q8k13", "EntityCanonicalName" -> 
      "Screech::35hd7"|>, "Creak" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Onomatopoeia"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Onomatopoeia", "Source-ambiguous sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.000014222655674585639, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(12.758084084826951 - 10177.272930196197*x))^(-1)], 
     "AudioSetID" -> "/m/07qh7jl", "EntityCanonicalName" -> "Creak::77z5d"|>, 
   "Rustle" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Onomatopoeia"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Onomatopoeia", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0006659250567636347, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.593651963413363 - 85.3341693300351*x))^(-1)], 
     "AudioSetID" -> "/m/07qwyj0", "EntityCanonicalName" -> 
      "Rustle::8z54d"|>, "Whir" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Onomatopoeia"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Onomatopoeia", "Source-ambiguous sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.00007314508632644043, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(10.207350860076092 - 735.4539537191546*x))^(-1)], 
     "AudioSetID" -> "/m/07s34ls", "EntityCanonicalName" -> "Whir::cdpz9"|>, 
   "Clatter" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Onomatopoeia"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Onomatopoeia", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> 0.0006130980499723166, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.81330889555405 - 67.1211826711395*x))^(-1)], 
     "AudioSetID" -> "/m/07qmpdm", "EntityCanonicalName" -> 
      "Clatter::gk9p7"|>, "Sizzle" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Onomatopoeia"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Onomatopoeia", "Source-ambiguous sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0006049708181582677, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.07889670168185 - 12.042034313212591*x))^(-1)], 
     "AudioSetID" -> "/m/07p9k1k", "EntityCanonicalName" -> 
      "Sizzle::mdh97"|>, "Clicking" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Tick", "Clip-clop", "Clickety-clack"}, 
     "FirstParents" -> {"Onomatopoeia"}, "FlattenedChildren" -> 
      {"Clip-clop", "Tick", "Clickety-clack"}, "FlattenedParents" -> 
      {"Onomatopoeia", "Source-ambiguous sounds"}, "BottomDepth" -> 1, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0003169620407479085, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.841996508089727 - 32.90147805740124*x))^(-1)], 
     "AudioSetID" -> "/m/07qc9xj", "EntityCanonicalName" -> 
      "Clicking::q258b"|>, "Clickety-clack" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Clicking"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Clicking", "Onomatopoeia", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0009422509384412986, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.489301079450012 - 12.479853657120236*x))^(-1)], 
     "AudioSetID" -> "/m/07rwm0c", "EntityCanonicalName" -> 
      "Clickety-clack::7889y"|>, "Rumble" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Onomatopoeia"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Onomatopoeia", "Source-ambiguous sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.00018895813967663777, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.993722829136304 - 109.87413093009067*x))^(-1)], 
     "AudioSetID" -> "/m/07phhsh", "EntityCanonicalName" -> 
      "Rumble::yt244"|>, "Blare" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Onomatopoeia"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Onomatopoeia", "Source-ambiguous sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/07r_r9m", "EntityCanonicalName" -> "Blare::s9v64"|>, 
   "Plop" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Onomatopoeia"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Onomatopoeia", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00037486856742300717, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.890084452774074 - 16.36374831361111*x))^(-1)], 
     "AudioSetID" -> "/m/07qyrcz", "EntityCanonicalName" -> "Plop::p7n55"|>, 
   "Jingle, tinkle" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Onomatopoeia"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Onomatopoeia", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.000521158740075888, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.975902771895538 - 23.46379562140677*x))^(-1)], 
     "AudioSetID" -> "/m/07qfgpx", "EntityCanonicalName" -> 
      "JingleTinkle::hqx6s"|>, "Fizz" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Onomatopoeia"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Onomatopoeia", "Source-ambiguous sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/t/dd00118", "EntityCanonicalName" -> "Fizz::9c9ht"|>, 
   "Puff" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Onomatopoeia"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Onomatopoeia", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/07q34h3", "EntityCanonicalName" -> "Puff::8734k"|>, 
   "Hum" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Onomatopoeia"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Onomatopoeia", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0006125900979839386, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.61264174903827 - 15.302177816452748*x))^(-1)], 
     "AudioSetID" -> "/m/07rcgpl", "EntityCanonicalName" -> "Hum::5vfw2"|>, 
   "Zing" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Onomatopoeia"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Onomatopoeia", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00003860435111673245, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(10.491534173630503 - 1520.3423370914759*x))^(-1)], 
     "AudioSetID" -> "/m/07p78v5", "EntityCanonicalName" -> "Zing::gbz27"|>, 
   "Boing" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Onomatopoeia"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Onomatopoeia", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00016610030019962514, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.428722875089072 - 14.361148862087672*x))^(-1)], 
     "AudioSetID" -> "/t/dd00121", "EntityCanonicalName" -> "Boing::8gmvx"|>, 
   "Crunch" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Onomatopoeia"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Onomatopoeia", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00011174943744317287, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.615026438877353 - 33.787440648706244*x))^(-1)], 
     "AudioSetID" -> "/m/07s12q4", "EntityCanonicalName" -> 
      "Crunch::72qfx"|>, "Silence" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Source-ambiguous sounds"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Source-ambiguous sounds"}, "BottomDepth" -> 0, "TopDepth" -> 1, 
     "Restrictions" -> {}, "ClassPrior" -> 0.003670461068019851, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(7.6458844851556975 - 21.670467556362144*x))^(-1)], 
     "AudioSetID" -> "/m/028v0c", "EntityCanonicalName" -> 
      "Silence::k52w3"|>, "Other sourceless" -> 
    <|"DepthIndex" -> 0.3333333333333333, "FirstChildren" -> 
      {"Sine wave", "Sound effect", "Pulse", "Infrasound", 
       "Bass (frequency range)", "Ringing (of resonator)"}, 
     "FirstParents" -> {"Source-ambiguous sounds"}, 
     "FlattenedChildren" -> {"Sine wave", "Sound effect", "Pulse", 
       "Infrasound", "Bass (frequency range)", "Ringing (of resonator)", 
       "Harmonic", "Chirp tone"}, "FlattenedParents" -> 
      {"Source-ambiguous sounds"}, "BottomDepth" -> 2, "TopDepth" -> 1, 
     "Restrictions" -> {"abstract"}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/t/dd00122", "EntityCanonicalName" -> 
      "OtherSourceless::ct9y6"|>, "Sine wave" -> 
    <|"DepthIndex" -> 0.6666666666666666, "FirstChildren" -> 
      {"Harmonic", "Chirp tone"}, "FirstParents" -> {"Other sourceless"}, 
     "FlattenedChildren" -> {"Harmonic", "Chirp tone"}, 
     "FlattenedParents" -> {"Other sourceless", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 1, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00016559234821124708, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.397872724676736 - 30.511161501248676*x))^(-1)], 
     "AudioSetID" -> "/m/01v_m0", "EntityCanonicalName" -> 
      "SineWave::q984w"|>, "Harmonic" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Sine wave"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Sine wave", "Other sourceless", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> 0.00031391432881764015, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.523560835387931 - 226.39769169658513*x))^(-1)], 
     "AudioSetID" -> "/m/0b9m1", "EntityCanonicalName" -> 
      "Harmonic::7zh34"|>, "Chirp tone" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Sine wave"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Sine wave", "Other sourceless", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00005435086275645226, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(10.026050614091378 - 104.81498512411021*x))^(-1)], 
     "AudioSetID" -> "/m/0hdsk", "EntityCanonicalName" -> 
      "ChirpTone::h74np"|>, "Sound effect" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Other sourceless"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Other sourceless", "Source-ambiguous sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.004027043363861248, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.037512683813686 - 18.63730853868479*x))^(-1)], 
     "AudioSetID" -> "/m/0c1dj", "EntityCanonicalName" -> 
      "SoundEffect::58y48"|>, "Pulse" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Other sourceless"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Other sourceless", "Source-ambiguous sounds"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.00004266796702375692, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(10.70936775236106 - 1004.1645398545885*x))^(-1)], 
     "AudioSetID" -> "/m/07pt_g0", "EntityCanonicalName" -> "Pulse::d4wnv"|>, 
   "Infrasound" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Other sourceless"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Other sourceless", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/01fhw5", "EntityCanonicalName" -> 
      "Infrasound::2c32y"|>, "Bass (frequency range)" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Other sourceless"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Other sourceless", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/017gp", "EntityCanonicalName" -> 
      "BassFrequencyRange::8mhv2"|>, "Ringing (of resonator)" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Other sourceless"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Other sourceless", "Source-ambiguous sounds"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/05szwhw", "EntityCanonicalName" -> 
      "RingingOfResonator::9bt3d"|>, "Channel, environment and background" -> 
    <|"DepthIndex" -> 0., "FirstChildren" -> {"Acoustic environment", 
       "Noise", "Sound reproduction"}, "FirstParents" -> {}, 
     "FlattenedChildren" -> {"Acoustic environment", "Noise", 
       "Sound reproduction", "Inside, small room", 
       "Inside, large room or hall", "Inside, public space", 
       "Outside, urban or manmade", "Outside, rural or natural", 
       "Reverberation", "Echo", "Hubbub, speech noise, speech babble", 
       "Background noise", "Cacophony", "White noise", "Pink noise", 
       "Throbbing", "Vibration", "Television", "Radio", "Loudspeaker", 
       "Headphones", "Recording", "Gramophone record", "Compact disc", "MP3", 
       "Environmental noise", "Tape hiss", "Static", "Mains hum", 
       "Distortion", "Sidetone", "Field recording"}, 
     "FlattenedParents" -> {}, "BottomDepth" -> 3, "TopDepth" -> 0, 
     "Restrictions" -> {"abstract"}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/t/dd00123", "EntityCanonicalName" -> 
      "ChannelEnvironmentAndBackground::bh585"|>, 
   "Acoustic environment" -> <|"DepthIndex" -> 0.5, 
     "FirstChildren" -> {"Inside, small room", "Inside, large room or hall", 
       "Inside, public space", "Outside, urban or manmade", 
       "Outside, rural or natural", "Reverberation", "Echo"}, 
     "FirstParents" -> {"Channel, environment and background"}, 
     "FlattenedChildren" -> {"Inside, small room", 
       "Inside, large room or hall", "Inside, public space", 
       "Outside, urban or manmade", "Outside, rural or natural", 
       "Reverberation", "Echo"}, "FlattenedParents" -> 
      {"Channel, environment and background"}, "BottomDepth" -> 1, 
     "TopDepth" -> 1, "Restrictions" -> {"abstract"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/t/dd00093", "EntityCanonicalName" -> 
      "AcousticEnvironment::mqc7v"|>, "Inside, small room" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Acoustic environment"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Acoustic environment", 
       "Channel, environment and background"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.03658930557883669, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(4.6850418742926525 - 9.410986259587446*x))^(-1)], 
     "AudioSetID" -> "/t/dd00125", "EntityCanonicalName" -> 
      "InsideSmallRoom::ng4kb"|>, "Inside, large room or hall" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Acoustic environment"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Acoustic environment", 
       "Channel, environment and background"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.013576540745368747, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(5.13166537851912 - 20.726452924055987*x))^(-1)], 
     "AudioSetID" -> "/t/dd00126", "EntityCanonicalName" -> 
      "InsideLargeRoomOrHall::qtv26"|>, "Inside, public space" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Acoustic environment"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Acoustic environment", 
       "Channel, environment and background"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0032696869491895624, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.447860535816931 - 34.60683888117298*x))^(-1)], 
     "AudioSetID" -> "/t/dd00127", "EntityCanonicalName" -> 
      "InsidePublicSpace::986py"|>, "Outside, urban or manmade" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Acoustic environment"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Acoustic environment", 
       "Channel, environment and background"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.016644062803183843, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(4.958109109011746 - 8.790135859857978*x))^(-1)], 
     "AudioSetID" -> "/t/dd00128", "EntityCanonicalName" -> 
      "OutsideUrbanOrManmade::dzv36"|>, "Outside, rural or natural" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Acoustic environment"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Acoustic environment", 
       "Channel, environment and background"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.01728611411649371, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(5.190131783553751 - 12.667338386554592*x))^(-1)], 
     "AudioSetID" -> "/t/dd00129", "EntityCanonicalName" -> 
      "OutsideRuralOrNatural::b62wh"|>, "Reverberation" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Acoustic environment"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Acoustic environment", 
       "Channel, environment and background"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.00036775723958571436, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.481526923644719 - 59.91069597505352*x))^(-1)], 
     "AudioSetID" -> "/m/01b9nn", "EntityCanonicalName" -> 
      "Reverberation::r66w9"|>, "Echo" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Acoustic environment"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Acoustic environment", "Channel, environment and background"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0005800811707277428, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.905718889982421 - 47.68902311978155*x))^(-1)], 
     "AudioSetID" -> "/m/01jnbd", "EntityCanonicalName" -> "Echo::7v985"|>, 
   "Noise" -> <|"DepthIndex" -> 0.3333333333333333, 
     "FirstChildren" -> {"Background noise", 
       "Hubbub, speech noise, speech babble", "Cacophony", "White noise", 
       "Pink noise", "Throbbing", "Vibration"}, "FirstParents" -> 
      {"Channel, environment and background"}, "FlattenedChildren" -> 
      {"Hubbub, speech noise, speech babble", "Background noise", 
       "Cacophony", "White noise", "Pink noise", "Throbbing", "Vibration", 
       "Environmental noise", "Tape hiss", "Static", "Mains hum", 
       "Distortion", "Sidetone"}, "FlattenedParents" -> 
      {"Channel, environment and background"}, "BottomDepth" -> 2, 
     "TopDepth" -> 1, "Restrictions" -> {}, "ClassPrior" -> 
      0.00011378124539668511, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(9.494183832399063 - 169.52601904330407*x))^(-1)], 
     "AudioSetID" -> "/m/096m7z", "EntityCanonicalName" -> "Noise::fs2rn"|>, 
   "Background noise" -> <|"DepthIndex" -> 0.6666666666666666, 
     "FirstChildren" -> {"Environmental noise", "Tape hiss", "Static", 
       "Mains hum", "Distortion", "Sidetone"}, "FirstParents" -> {"Noise"}, 
     "FlattenedChildren" -> {"Environmental noise", "Tape hiss", "Static", 
       "Mains hum", "Distortion", "Sidetone"}, "FlattenedParents" -> 
      {"Noise", "Channel, environment and background"}, "BottomDepth" -> 1, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> Missing[], 
     "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/093_4n", "EntityCanonicalName" -> 
      "BackgroundNoise::847j4"|>, "Environmental noise" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Background noise"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Background noise", "Noise", 
       "Channel, environment and background"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {}, "ClassPrior" -> 
      0.0005546835713088399, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.672592617990016 - 22.02756627308125*x))^(-1)], 
     "AudioSetID" -> "/m/06_y0by", "EntityCanonicalName" -> 
      "EnvironmentalNoise::n49sv"|>, "Tape hiss" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Background noise"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Background noise", "Noise", 
       "Channel, environment and background"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/09d1b1", "EntityCanonicalName" -> 
      "TapeHiss::dw54c"|>, "Static" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Background noise"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Background noise", "Noise", "Channel, environment and background"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00020114898739771118, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.472506341696228 - 59.32202339835651*x))^(-1)], 
     "AudioSetID" -> "/m/07rgkc5", "EntityCanonicalName" -> 
      "Static::hqd4b"|>, "Mains hum" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Background noise"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Background noise", "Noise", "Channel, environment and background"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.00015390945247855173, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(8.874745629323836 - 21.45947251905717*x))^(-1)], 
     "AudioSetID" -> "/m/06xkwv", "EntityCanonicalName" -> 
      "MainsHum::h9f76"|>, "Distortion" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Background noise"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Background noise", "Noise", "Channel, environment and background"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {}, 
     "ClassPrior" -> 0.0013658828967485993, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.898404101069834 - 13.253229706570407*x))^(-1)], 
     "AudioSetID" -> "/m/0g12c5", "EntityCanonicalName" -> 
      "Distortion::xwk4m"|>, "Sidetone" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Background noise"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Background noise", "Noise", "Channel, environment and background"}, 
     "BottomDepth" -> 0, "TopDepth" -> 3, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> 0.00003758844713997633, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(10.129477372744992 - 45.64578826263089*x))^(-1)], 
     "AudioSetID" -> "/m/08p9q4", "EntityCanonicalName" -> 
      "Sidetone::tqyb4"|>, "Cacophony" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Noise"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Noise", "Channel, environment and background"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.000364201575667068, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.095691448961347 - 51.21706118853772*x))^(-1)], 
     "AudioSetID" -> "/m/07szfh9", "EntityCanonicalName" -> 
      "Cacophony::5hmjz"|>, "White noise" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Noise"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Noise", "Channel, environment and background"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0008208504132189425, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.247233051896756 - 25.150267491258774*x))^(-1)], 
     "AudioSetID" -> "/m/0chx_", "EntityCanonicalName" -> 
      "WhiteNoise::ymv98"|>, "Pink noise" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Noise"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Noise", "Channel, environment and background"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.00043023533415621556, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.569491776740099 - 26.67936143719125*x))^(-1)], 
     "AudioSetID" -> "/m/0cj0r", "EntityCanonicalName" -> 
      "PinkNoise::f43p3"|>, "Throbbing" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Noise"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Noise", "Channel, environment and background"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.0005755096028323402, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(8.001000242209807 - 14.07488566356486*x))^(-1)], 
     "AudioSetID" -> "/m/07p_0gm", "EntityCanonicalName" -> 
      "Throbbing::3tbjn"|>, "Vibration" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Noise"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Noise", "Channel, environment and background"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> 0.0011515271576530586, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.542175095080201 - 21.825822868024957*x))^(-1)], 
     "AudioSetID" -> "/m/01jwx6", "EntityCanonicalName" -> 
      "Vibration::y67qm"|>, "Sound reproduction" -> 
    <|"DepthIndex" -> 0.3333333333333333, "FirstChildren" -> 
      {"Television", "Radio", "Loudspeaker", "Headphones", "Recording", 
       "Gramophone record", "Compact disc", "MP3"}, 
     "FirstParents" -> {"Channel, environment and background"}, 
     "FlattenedChildren" -> {"Television", "Radio", "Loudspeaker", 
       "Headphones", "Recording", "Gramophone record", "Compact disc", "MP3", 
       "Field recording"}, "FlattenedParents" -> 
      {"Channel, environment and background"}, "BottomDepth" -> 2, 
     "TopDepth" -> 1, "Restrictions" -> {"abstract"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/07bm98", "EntityCanonicalName" -> 
      "SoundReproduction::qvz5z"|>, "Television" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Sound reproduction"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Sound reproduction", 
       "Channel, environment and background"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {}, "ClassPrior" -> 
      0.001006252888976934, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(7.092940803232753 - 23.387596309830716*x))^(-1)], 
     "AudioSetID" -> "/m/07c52", "EntityCanonicalName" -> 
      "Television::ykdn4"|>, "Radio" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Sound reproduction"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Sound reproduction", "Channel, environment and background"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {}, 
     "ClassPrior" -> 0.002022664817721429, "LogisticRegressionUnbalanced" -> 
      Function[x, (1 + E^(6.9402386016445154 - 9.323632740421738*x))^(-1)], 
     "AudioSetID" -> "/m/06bz3", "EntityCanonicalName" -> "Radio::96j6p"|>, 
   "Loudspeaker" -> <|"DepthIndex" -> 1., "FirstChildren" -> {}, 
     "FirstParents" -> {"Sound reproduction"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Sound reproduction", 
       "Channel, environment and background"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/0cfpc", "EntityCanonicalName" -> 
      "Loudspeaker::7tc7m"|>, "Headphones" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Sound reproduction"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Sound reproduction", "Channel, environment and background"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/01b7fy", "EntityCanonicalName" -> 
      "Headphones::8ryyq"|>, "Recording" -> 
    <|"DepthIndex" -> 0.6666666666666666, "FirstChildren" -> 
      {"Field recording"}, "FirstParents" -> {"Sound reproduction"}, 
     "FlattenedChildren" -> {"Field recording"}, "FlattenedParents" -> 
      {"Sound reproduction", "Channel, environment and background"}, 
     "BottomDepth" -> 1, "TopDepth" -> 2, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/025l19", "EntityCanonicalName" -> 
      "Recording::hn4fs"|>, "Field recording" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Recording"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Recording", "Sound reproduction", 
       "Channel, environment and background"}, "BottomDepth" -> 0, 
     "TopDepth" -> 3, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> 0.00012038462124559986, 
     "LogisticRegressionUnbalanced" -> Function[x, 
       (1 + E^(9.266304417067644 - 94.61242315522969*x))^(-1)], 
     "AudioSetID" -> "/m/07hvw1", "EntityCanonicalName" -> 
      "FieldRecording::4745n"|>, "Gramophone record" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Sound reproduction"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Sound reproduction", 
       "Channel, environment and background"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/0174nj", "EntityCanonicalName" -> 
      "GramophoneRecord::m9vsb"|>, "Compact disc" -> 
    <|"DepthIndex" -> 1., "FirstChildren" -> {}, "FirstParents" -> 
      {"Sound reproduction"}, "FlattenedChildren" -> {}, 
     "FlattenedParents" -> {"Sound reproduction", 
       "Channel, environment and background"}, "BottomDepth" -> 0, 
     "TopDepth" -> 2, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/01www", "EntityCanonicalName" -> 
      "CompactDisc::46vyp"|>, "MP3" -> <|"DepthIndex" -> 1., 
     "FirstChildren" -> {}, "FirstParents" -> {"Sound reproduction"}, 
     "FlattenedChildren" -> {}, "FlattenedParents" -> 
      {"Sound reproduction", "Channel, environment and background"}, 
     "BottomDepth" -> 0, "TopDepth" -> 2, "Restrictions" -> {"blacklist"}, 
     "ClassPrior" -> Missing[], "LogisticRegressionUnbalanced" -> Missing[], 
     "AudioSetID" -> "/m/04zc0", "EntityCanonicalName" -> "MP3::f3yz8"|>|>, 
 "Graph" -> Graph[{"Human sounds", "Human voice", "Speech", 
    "Male speech, man speaking", "Female speech, woman speaking", 
    "Child speech, kid speaking", "Conversation", "Narration, monologue", 
    "Babbling", "Speech synthesizer", "Shout", "Bellow", "Whoop", "Yell", 
    "Battle cry", "Children shouting", "Screaming", "Whispering", "Laughter", 
    "Baby laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle", 
    "Crying, sobbing", "Baby cry, infant cry", "Whimper", "Wail, moan", 
    "Sigh", "Singing", "Choir", "Yodeling", "Chant", "Mantra", 
    "Male singing", "Female singing", "Child singing", "Synthetic singing", 
    "Rapping", "Humming", "Groan", "Grunt", "Yawn", "Whistling", 
    "Wolf-whistling", "Respiratory sounds", "Breathing", "Wheeze", "Snoring", 
    "Gasp", "Pant", "Snort", "Cough", "Throat clearing", "Sneeze", "Sniff", 
    "Human locomotion", "Run", "Shuffle", "Walk, footsteps", "Digestive", 
    "Chewing, mastication", "Biting", "Gargling", "Stomach rumble", 
    "Burping, eructation", "Hiccup", "Fart", "Hands", "Finger snapping", 
    "Clapping", "Heart sounds, heartbeat", "Heart murmur", 
    "Otoacoustic emission", "Tinnitus, ringing in the ears", 
    "Human group actions", "Cheering", "Applause", "Chatter", "Crowd", 
    "Hubbub, speech noise, speech babble", "Booing", "Children playing", 
    "Animal", "Domestic animals, pets", "Dog", "Bark", "Yip", "Howl", 
    "Bow-wow", "Growling", "Whimper (dog)", "Bay", "Cat", "Purr", "Meow", 
    "Hiss", "Cat communication", "Caterwaul", 
    "Livestock, farm animals, working animals", "Horse", "Clip-clop", 
    "Neigh, whinny", "Snort (horse)", "Nicker", "Donkey, ass", 
    "Cattle, bovinae", "Moo", "Cowbell", "Yak", "Pig", "Oink", "Goat", 
    "Bleat", "Sheep", "Fowl", "Chicken, rooster", "Cluck", 
    "Crowing, cock-a-doodle-doo", "Turkey", "Gobble", "Duck", "Quack", 
    "Goose", "Honk", "Wild animals", "Roaring cats (lions, tigers)", "Roar", 
    "Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet", 
    "Squawk", "Pigeon, dove", "Coo", "Crow", "Caw", "Owl", "Hoot", 
    "Gull, seagull", "Bird flight, flapping wings", "Canidae, dogs, wolves", 
    "Rodents, rats, mice", "Mouse", "Chipmunk", "Patter", "Insect", 
    "Cricket", "Mosquito", "Fly, housefly", "Buzz", "Bee, wasp, etc.", 
    "Frog", "Croak", "Snake", "Rattle", "Whale vocalization", "Music", 
    "Musical instrument", "Plucked string instrument", "Guitar", 
    "Electric guitar", "Bass guitar", "Acoustic guitar", 
    "Steel guitar, slide guitar", "Tapping (guitar technique)", "Strum", 
    "Banjo", "Sitar", "Mandolin", "Zither", "Ukulele", "Keyboard (musical)", 
    "Piano", "Electric piano", "Clavinet", "Rhodes piano", "Organ", 
    "Electronic organ", "Hammond organ", "Synthesizer", "Sampler", 
    "Mellotron", "Harpsichord", "Percussion", "Drum kit", "Drum machine", 
    "Drum", "Snare drum", "Rimshot", "Drum roll", "Bass drum", "Timpani", 
    "Tabla", "Cymbal", "Hi-hat", "Crash cymbal", "Wood block", "Tambourine", 
    "Rattle (instrument)", "Maraca", "Gong", "Tubular bells", 
    "Mallet percussion", "Marimba, xylophone", "Glockenspiel", "Vibraphone", 
    "Steelpan", "Orchestra", "Brass instrument", "French horn", "Trumpet", 
    "Trombone", "Cornet", "Bugle", "Bowed string instrument", 
    "String section", "Violin, fiddle", "Pizzicato", "Cello", "Double bass", 
    "Wind instrument, woodwind instrument", "Flute", "Saxophone", 
    "Alto saxophone", "Soprano saxophone", "Clarinet", "Oboe", "Bassoon", 
    "Harp", "Bell", "Church bell", "Jingle bell", "Bicycle bell", 
    "Tuning fork", "Chime", "Wind chime", "Change ringing (campanology)", 
    "Harmonica", "Accordion", "Bagpipes", "Didgeridoo", "Shofar", "Theremin", 
    "Singing bowl", "Musical ensemble", "Bass (instrument role)", 
    "Scratching (performance technique)", "Music genre", "Pop music", 
    "Hip hop music", "Grime music", "Trap music", "Beatboxing", "Rock music", 
    "Heavy metal", "Punk rock", "Grunge", "Progressive rock", 
    "Rock and roll", "Psychedelic rock", "Rhythm and blues", "Soul music", 
    "Reggae", "Dub", "Country", "Swing music", "Bluegrass", "Funk", 
    "Folk music", "Middle Eastern music", "Jazz", "Disco", "Classical music", 
    "Opera", "Electronic music", "House music", "Techno", "Dubstep", 
    "Electro", "Drum and bass", "Oldschool jungle", "Electronica", 
    "Electronic dance music", "Ambient music", "Drone music", "Trance music", 
    "Noise music", "UK garage", "Music of Latin America", "Cumbia", 
    "Salsa music", "Soca music", "Kuduro", "Funk carioca", "Flamenco", 
    "Blues", "Music for children", "New-age music", "Vocal music", 
    "A capella", "Music of Africa", "Afrobeat", "Kwaito", "Christian music", 
    "Gospel music", "Music of Asia", "Carnatic music", "Music of Bollywood", 
    "Ska", "Traditional music", "Independent music", "Musical concepts", 
    "Song", "Melody", "Musical note", "Beat", "Drum beat", "Chord", 
    "Harmony", "Bassline", "Loop", "Drone", "Music role", "Background music", 
    "Theme music", "Jingle (music)", "Soundtrack music", "Lullaby", 
    "Video game music", "Christmas music", "Dance music", "Wedding music", 
    "Birthday music", "Music mood", "Happy music", "Funny music", 
    "Sad music", "Tender music", "Exciting music", "Angry music", 
    "Scary music", "Natural sounds", "Wind", "Howl (wind)", 
    "Rustling leaves", "Wind noise (microphone)", "Thunderstorm", "Thunder", 
    "Water", "Rain", "Raindrop", "Rain on surface", "Stream", "Waterfall", 
    "Ocean", "Waves, surf", "Steam", "Gurgling", "Fire", "Crackle", 
    "Wildfire", "Sounds of things", "Vehicle", "Boat, Water vehicle", 
    "Sailboat, sailing ship", "Rowboat, canoe, kayak", 
    "Motorboat, speedboat", "Ship", "Motor vehicle (road)", "Car", 
    "Vehicle horn, car horn, honking", "Toot", "Car alarm", 
    "Power windows, electric windows", "Skidding", "Tire squeal", 
    "Car passing by", "Race car, auto racing", "Truck", "Air brake", 
    "Air horn, truck horn", "Reversing beeps", 
    "Ice cream truck, ice cream van", "Bus", "Emergency vehicle", 
    "Police car (siren)", "Ambulance (siren)", 
    "Fire engine, fire truck (siren)", "Motorcycle", 
    "Traffic noise, roadway noise", "Rail transport", "Train", 
    "Train whistle", "Train horn", "Railroad car, train wagon", 
    "Train wheels squealing", "Subway, metro, underground", "Aircraft", 
    "Aircraft engine", "Jet engine", "Propeller, airscrew", "Helicopter", 
    "Fixed-wing aircraft, airplane", "Non-motorized land vehicle", "Bicycle", 
    "Skateboard", "Engine", "Light engine (high frequency)", 
    "Dental drill, dentist's drill", "Lawn mower", "Chainsaw", 
    "Medium engine (mid frequency)", "Heavy engine (low frequency)", 
    "Engine knocking", "Engine starting", "Idling", 
    "Accelerating, revving, vroom", "Domestic sounds, home sounds", "Door", 
    "Doorbell", "Ding-dong", "Sliding door", "Slam", "Knock", "Tap", 
    "Squeak", "Cupboard open or close", "Drawer open or close", 
    "Dishes, pots, and pans", "Cutlery, silverware", "Chopping (food)", 
    "Frying (food)", "Microwave oven", "Blender", "Kettle whistle", 
    "Water tap, faucet", "Sink (filling or washing)", 
    "Bathtub (filling or washing)", "Hair dryer", "Toilet flush", 
    "Toothbrush", "Electric toothbrush", "Vacuum cleaner", 
    "Zipper (clothing)", "Velcro, hook and loop fastener", "Keys jangling", 
    "Coin (dropping)", "Packing tape, duct tape", "Scissors", 
    "Electric shaver, electric razor", "Shuffling cards", "Typing", 
    "Typewriter", "Computer keyboard", "Writing", "Alarm", "Telephone", 
    "Telephone bell ringing", "Ringtone", "Cellphone buzz, vibrating alert", 
    "Telephone dialing, DTMF", "Dial tone", "Busy signal", "Alarm clock", 
    "Siren", "Civil defense siren", "Buzzer", "Smoke detector, smoke alarm", 
    "Fire alarm", "Foghorn", "Whistle", "Steam whistle", "Mechanisms", 
    "Ratchet, pawl", "Clock", "Tick", "Tick-tock", "Gears", "Pulleys", 
    "Sewing machine", "Mechanical fan", "Air conditioning", "Cash register", 
    "Printer", "Camera", "Single-lens reflex camera", "Tools", "Hammer", 
    "Jackhammer", "Sawing", "Filing (rasp)", "Sanding", "Power tool", 
    "Drill", "Explosion", "Gunshot, gunfire", "Machine gun", "Fusillade", 
    "Artillery fire", "Cap gun", "Fireworks", "Firecracker", "Burst, pop", 
    "Eruption", "Boom", "Sonic boom", "Wood", "Chop", "Splinter", "Crack", 
    "Snap", "Glass", "Chink, clink", "Shatter", "Liquid", "Splash, splatter", 
    "Slosh", "Squish", "Drip", "Pour", "Trickle, dribble", "Gush", 
    "Fill (with liquid)", "Spray", "Pump (liquid)", "Stir", "Boiling", 
    "Miscellaneous sources", "Sonar", "Duck call (hunting tool)", "Arrow", 
    "Whoosh, swoosh, swish", "Thump, thud", "Thunk", "Clunk", "Wobble", 
    "Sound equipment", "Microphone", "Electronic tuner", "Guitar amplifier", 
    "Effects unit", "Chorus effect", "Specific impact sounds", 
    "Basketball bounce", "Source-ambiguous sounds", "Generic impact sounds", 
    "Bang", "Slap, smack", "Whack, thwack", "Smash, crash", "Breaking", 
    "Bouncing", "Whip", "Flap", "Surface contact", "Scratch", "Scrape", 
    "Rub", "Roll", "Grind", "Deformable shell", "Crushing", 
    "Crumpling, crinkling", "Tearing", "Onomatopoeia", "Brief tone", 
    "Beep, bleep", "Ping", "Ding", "Clang", "Twang", "Squeal", "Screech", 
    "Creak", "Rustle", "Whir", "Clatter", "Sizzle", "Clicking", 
    "Clickety-clack", "Rumble", "Blare", "Plop", "Jingle, tinkle", "Fizz", 
    "Puff", "Hum", "Zing", "Boing", "Crunch", "Silence", "Other sourceless", 
    "Sine wave", "Harmonic", "Chirp tone", "Sound effect", "Pulse", 
    "Infrasound", "Bass (frequency range)", "Ringing (of resonator)", 
    "Channel, environment and background", "Acoustic environment", 
    "Inside, small room", "Inside, large room or hall", 
    "Inside, public space", "Outside, urban or manmade", 
    "Outside, rural or natural", "Reverberation", "Echo", "Noise", 
    "Background noise", "Environmental noise", "Tape hiss", "Static", 
    "Mains hum", "Distortion", "Sidetone", "Cacophony", "White noise", 
    "Pink noise", "Throbbing", "Vibration", "Sound reproduction", 
    "Television", "Radio", "Loudspeaker", "Headphones", "Recording", 
    "Field recording", "Gramophone record", "Compact disc", "MP3"}, 
   {DirectedEdge["Human sounds", "Human voice"], DirectedEdge["Human sounds", 
     "Whistling"], DirectedEdge["Human sounds", "Respiratory sounds"], 
    DirectedEdge["Human sounds", "Human locomotion"], 
    DirectedEdge["Human sounds", "Digestive"], DirectedEdge["Human sounds", 
     "Hands"], DirectedEdge["Human sounds", "Heart sounds, heartbeat"], 
    DirectedEdge["Human sounds", "Otoacoustic emission"], 
    DirectedEdge["Human sounds", "Human group actions"], 
    DirectedEdge["Human voice", "Speech"], DirectedEdge["Human voice", 
     "Shout"], DirectedEdge["Human voice", "Screaming"], 
    DirectedEdge["Human voice", "Whispering"], DirectedEdge["Human voice", 
     "Laughter"], DirectedEdge["Human voice", "Crying, sobbing"], 
    DirectedEdge["Human voice", "Wail, moan"], DirectedEdge["Human voice", 
     "Sigh"], DirectedEdge["Human voice", "Singing"], 
    DirectedEdge["Human voice", "Humming"], DirectedEdge["Human voice", 
     "Groan"], DirectedEdge["Human voice", "Grunt"], 
    DirectedEdge["Human voice", "Yawn"], DirectedEdge["Speech", 
     "Male speech, man speaking"], DirectedEdge["Speech", 
     "Female speech, woman speaking"], DirectedEdge["Speech", 
     "Child speech, kid speaking"], DirectedEdge["Speech", "Conversation"], 
    DirectedEdge["Speech", "Narration, monologue"], 
    DirectedEdge["Speech", "Babbling"], DirectedEdge["Speech", 
     "Speech synthesizer"], DirectedEdge["Shout", "Bellow"], 
    DirectedEdge["Shout", "Whoop"], DirectedEdge["Shout", "Yell"], 
    DirectedEdge["Shout", "Battle cry"], DirectedEdge["Shout", 
     "Children shouting"], DirectedEdge["Laughter", "Baby laughter"], 
    DirectedEdge["Laughter", "Giggle"], DirectedEdge["Laughter", "Snicker"], 
    DirectedEdge["Laughter", "Belly laugh"], DirectedEdge["Laughter", 
     "Chuckle, chortle"], DirectedEdge["Crying, sobbing", 
     "Baby cry, infant cry"], DirectedEdge["Crying, sobbing", "Whimper"], 
    DirectedEdge["Singing", "Choir"], DirectedEdge["Singing", "Yodeling"], 
    DirectedEdge["Singing", "Chant"], DirectedEdge["Singing", 
     "Male singing"], DirectedEdge["Singing", "Female singing"], 
    DirectedEdge["Singing", "Child singing"], DirectedEdge["Singing", 
     "Synthetic singing"], DirectedEdge["Singing", "Rapping"], 
    DirectedEdge["Chant", "Mantra"], DirectedEdge["Whistling", 
     "Wolf-whistling"], DirectedEdge["Respiratory sounds", "Breathing"], 
    DirectedEdge["Respiratory sounds", "Cough"], 
    DirectedEdge["Respiratory sounds", "Sneeze"], 
    DirectedEdge["Respiratory sounds", "Sniff"], DirectedEdge["Breathing", 
     "Wheeze"], DirectedEdge["Breathing", "Snoring"], 
    DirectedEdge["Breathing", "Gasp"], DirectedEdge["Breathing", "Pant"], 
    DirectedEdge["Breathing", "Snort"], DirectedEdge["Cough", 
     "Throat clearing"], DirectedEdge["Human locomotion", "Run"], 
    DirectedEdge["Human locomotion", "Shuffle"], 
    DirectedEdge["Human locomotion", "Walk, footsteps"], 
    DirectedEdge["Digestive", "Chewing, mastication"], 
    DirectedEdge["Digestive", "Biting"], DirectedEdge["Digestive", 
     "Gargling"], DirectedEdge["Digestive", "Stomach rumble"], 
    DirectedEdge["Digestive", "Burping, eructation"], 
    DirectedEdge["Digestive", "Hiccup"], DirectedEdge["Digestive", "Fart"], 
    DirectedEdge["Hands", "Finger snapping"], DirectedEdge["Hands", 
     "Clapping"], DirectedEdge["Heart sounds, heartbeat", "Heart murmur"], 
    DirectedEdge["Otoacoustic emission", "Tinnitus, ringing in the ears"], 
    DirectedEdge["Human group actions", "Clapping"], 
    DirectedEdge["Human group actions", "Cheering"], 
    DirectedEdge["Human group actions", "Applause"], 
    DirectedEdge["Human group actions", "Chatter"], 
    DirectedEdge["Human group actions", "Crowd"], 
    DirectedEdge["Human group actions", 
     "Hubbub, speech noise, speech babble"], 
    DirectedEdge["Human group actions", "Booing"], 
    DirectedEdge["Human group actions", "Children playing"], 
    DirectedEdge["Human group actions", "Children shouting"], 
    DirectedEdge["Animal", "Domestic animals, pets"], 
    DirectedEdge["Animal", "Livestock, farm animals, working animals"], 
    DirectedEdge["Animal", "Wild animals"], 
    DirectedEdge["Domestic animals, pets", "Dog"], 
    DirectedEdge["Domestic animals, pets", "Cat"], 
    DirectedEdge["Dog", "Bark"], DirectedEdge["Dog", "Yip"], 
    DirectedEdge["Dog", "Howl"], DirectedEdge["Dog", "Bow-wow"], 
    DirectedEdge["Dog", "Growling"], DirectedEdge["Dog", "Whimper (dog)"], 
    DirectedEdge["Dog", "Bay"], DirectedEdge["Cat", "Purr"], 
    DirectedEdge["Cat", "Meow"], DirectedEdge["Cat", "Hiss"], 
    DirectedEdge["Cat", "Cat communication"], DirectedEdge["Cat", 
     "Caterwaul"], DirectedEdge["Cat", "Growling"], 
    DirectedEdge["Livestock, farm animals, working animals", "Horse"], 
    DirectedEdge["Livestock, farm animals, working animals", "Donkey, ass"], 
    DirectedEdge["Livestock, farm animals, working animals", 
     "Cattle, bovinae"], DirectedEdge[
     "Livestock, farm animals, working animals", "Pig"], 
    DirectedEdge["Livestock, farm animals, working animals", "Goat"], 
    DirectedEdge["Livestock, farm animals, working animals", "Sheep"], 
    DirectedEdge["Livestock, farm animals, working animals", "Fowl"], 
    DirectedEdge["Horse", "Clip-clop"], DirectedEdge["Horse", 
     "Neigh, whinny"], DirectedEdge["Horse", "Snort (horse)"], 
    DirectedEdge["Horse", "Nicker"], DirectedEdge["Cattle, bovinae", "Moo"], 
    DirectedEdge["Cattle, bovinae", "Cowbell"], 
    DirectedEdge["Cattle, bovinae", "Yak"], DirectedEdge["Pig", "Oink"], 
    DirectedEdge["Goat", "Bleat"], DirectedEdge["Sheep", "Bleat"], 
    DirectedEdge["Fowl", "Chicken, rooster"], DirectedEdge["Fowl", "Turkey"], 
    DirectedEdge["Fowl", "Duck"], DirectedEdge["Fowl", "Goose"], 
    DirectedEdge["Chicken, rooster", "Cluck"], 
    DirectedEdge["Chicken, rooster", "Crowing, cock-a-doodle-doo"], 
    DirectedEdge["Turkey", "Gobble"], DirectedEdge["Duck", "Quack"], 
    DirectedEdge["Goose", "Honk"], DirectedEdge["Wild animals", 
     "Roaring cats (lions, tigers)"], DirectedEdge["Wild animals", "Bird"], 
    DirectedEdge["Wild animals", "Canidae, dogs, wolves"], 
    DirectedEdge["Wild animals", "Rodents, rats, mice"], 
    DirectedEdge["Wild animals", "Insect"], DirectedEdge["Wild animals", 
     "Frog"], DirectedEdge["Wild animals", "Snake"], 
    DirectedEdge["Wild animals", "Whale vocalization"], 
    DirectedEdge["Roaring cats (lions, tigers)", "Roar"], 
    DirectedEdge["Roaring cats (lions, tigers)", "Growling"], 
    DirectedEdge["Bird", "Bird vocalization, bird call, bird song"], 
    DirectedEdge["Bird", "Pigeon, dove"], DirectedEdge["Bird", "Crow"], 
    DirectedEdge["Bird", "Owl"], DirectedEdge["Bird", "Gull, seagull"], 
    DirectedEdge["Bird", "Bird flight, flapping wings"], 
    DirectedEdge["Bird vocalization, bird call, bird song", "Chirp, tweet"], 
    DirectedEdge["Bird vocalization, bird call, bird song", "Squawk"], 
    DirectedEdge["Pigeon, dove", "Coo"], DirectedEdge["Crow", "Caw"], 
    DirectedEdge["Owl", "Hoot"], DirectedEdge["Canidae, dogs, wolves", 
     "Howl"], DirectedEdge["Canidae, dogs, wolves", "Growling"], 
    DirectedEdge["Rodents, rats, mice", "Mouse"], 
    DirectedEdge["Rodents, rats, mice", "Chipmunk"], 
    DirectedEdge["Rodents, rats, mice", "Patter"], 
    DirectedEdge["Insect", "Cricket"], DirectedEdge["Insect", "Mosquito"], 
    DirectedEdge["Insect", "Fly, housefly"], DirectedEdge["Insect", 
     "Bee, wasp, etc."], DirectedEdge["Fly, housefly", "Buzz"], 
    DirectedEdge["Bee, wasp, etc.", "Buzz"], DirectedEdge["Frog", "Croak"], 
    DirectedEdge["Snake", "Hiss"], DirectedEdge["Snake", "Rattle"], 
    DirectedEdge["Music", "Musical instrument"], DirectedEdge["Music", 
     "Music genre"], DirectedEdge["Music", "Musical concepts"], 
    DirectedEdge["Music", "Music role"], DirectedEdge["Music", "Music mood"], 
    DirectedEdge["Musical instrument", "Plucked string instrument"], 
    DirectedEdge["Musical instrument", "Keyboard (musical)"], 
    DirectedEdge["Musical instrument", "Percussion"], 
    DirectedEdge["Musical instrument", "Orchestra"], 
    DirectedEdge["Musical instrument", "Brass instrument"], 
    DirectedEdge["Musical instrument", "Bowed string instrument"], 
    DirectedEdge["Musical instrument", 
     "Wind instrument, woodwind instrument"], 
    DirectedEdge["Musical instrument", "Harp"], 
    DirectedEdge["Musical instrument", "Choir"], 
    DirectedEdge["Musical instrument", "Bell"], 
    DirectedEdge["Musical instrument", "Harmonica"], 
    DirectedEdge["Musical instrument", "Accordion"], 
    DirectedEdge["Musical instrument", "Bagpipes"], 
    DirectedEdge["Musical instrument", "Didgeridoo"], 
    DirectedEdge["Musical instrument", "Shofar"], 
    DirectedEdge["Musical instrument", "Theremin"], 
    DirectedEdge["Musical instrument", "Singing bowl"], 
    DirectedEdge["Musical instrument", "Musical ensemble"], 
    DirectedEdge["Musical instrument", "Bass (instrument role)"], 
    DirectedEdge["Musical instrument", "Scratching (performance technique)"], 
    DirectedEdge["Plucked string instrument", "Guitar"], 
    DirectedEdge["Plucked string instrument", "Banjo"], 
    DirectedEdge["Plucked string instrument", "Sitar"], 
    DirectedEdge["Plucked string instrument", "Mandolin"], 
    DirectedEdge["Plucked string instrument", "Zither"], 
    DirectedEdge["Plucked string instrument", "Ukulele"], 
    DirectedEdge["Guitar", "Electric guitar"], DirectedEdge["Guitar", 
     "Bass guitar"], DirectedEdge["Guitar", "Acoustic guitar"], 
    DirectedEdge["Guitar", "Steel guitar, slide guitar"], 
    DirectedEdge["Guitar", "Tapping (guitar technique)"], 
    DirectedEdge["Guitar", "Strum"], DirectedEdge["Keyboard (musical)", 
     "Piano"], DirectedEdge["Keyboard (musical)", "Organ"], 
    DirectedEdge["Keyboard (musical)", "Synthesizer"], 
    DirectedEdge["Keyboard (musical)", "Harpsichord"], 
    DirectedEdge["Piano", "Electric piano"], DirectedEdge["Electric piano", 
     "Clavinet"], DirectedEdge["Electric piano", "Rhodes piano"], 
    DirectedEdge["Organ", "Electronic organ"], DirectedEdge["Organ", 
     "Hammond organ"], DirectedEdge["Synthesizer", "Sampler"], 
    DirectedEdge["Synthesizer", "Mellotron"], DirectedEdge["Percussion", 
     "Drum kit"], DirectedEdge["Percussion", "Drum"], 
    DirectedEdge["Percussion", "Cymbal"], DirectedEdge["Percussion", 
     "Cowbell"], DirectedEdge["Percussion", "Wood block"], 
    DirectedEdge["Percussion", "Tambourine"], DirectedEdge["Percussion", 
     "Rattle (instrument)"], DirectedEdge["Percussion", "Gong"], 
    DirectedEdge["Percussion", "Tubular bells"], DirectedEdge["Percussion", 
     "Mallet percussion"], DirectedEdge["Drum kit", "Drum machine"], 
    DirectedEdge["Drum", "Snare drum"], DirectedEdge["Drum", "Bass drum"], 
    DirectedEdge["Drum", "Timpani"], DirectedEdge["Drum", "Tabla"], 
    DirectedEdge["Snare drum", "Rimshot"], DirectedEdge["Snare drum", 
     "Drum roll"], DirectedEdge["Cymbal", "Hi-hat"], 
    DirectedEdge["Cymbal", "Crash cymbal"], 
    DirectedEdge["Rattle (instrument)", "Maraca"], 
    DirectedEdge["Mallet percussion", "Marimba, xylophone"], 
    DirectedEdge["Mallet percussion", "Glockenspiel"], 
    DirectedEdge["Mallet percussion", "Vibraphone"], 
    DirectedEdge["Mallet percussion", "Steelpan"], 
    DirectedEdge["Brass instrument", "French horn"], 
    DirectedEdge["Brass instrument", "Trumpet"], 
    DirectedEdge["Brass instrument", "Trombone"], 
    DirectedEdge["Brass instrument", "Cornet"], 
    DirectedEdge["Brass instrument", "Bugle"], 
    DirectedEdge["Bowed string instrument", "String section"], 
    DirectedEdge["Bowed string instrument", "Violin, fiddle"], 
    DirectedEdge["Bowed string instrument", "Cello"], 
    DirectedEdge["Bowed string instrument", "Double bass"], 
    DirectedEdge["Violin, fiddle", "Pizzicato"], 
    DirectedEdge["Wind instrument, woodwind instrument", "Flute"], 
    DirectedEdge["Wind instrument, woodwind instrument", "Saxophone"], 
    DirectedEdge["Wind instrument, woodwind instrument", "Clarinet"], 
    DirectedEdge["Wind instrument, woodwind instrument", "Oboe"], 
    DirectedEdge["Wind instrument, woodwind instrument", "Bassoon"], 
    DirectedEdge["Saxophone", "Alto saxophone"], DirectedEdge["Saxophone", 
     "Soprano saxophone"], DirectedEdge["Bell", "Church bell"], 
    DirectedEdge["Bell", "Cowbell"], DirectedEdge["Bell", "Jingle bell"], 
    DirectedEdge["Bell", "Bicycle bell"], DirectedEdge["Bell", 
     "Tuning fork"], DirectedEdge["Bell", "Chime"], 
    DirectedEdge["Bell", "Change ringing (campanology)"], 
    DirectedEdge["Chime", "Wind chime"], DirectedEdge["Music genre", 
     "Pop music"], DirectedEdge["Music genre", "Hip hop music"], 
    DirectedEdge["Music genre", "Rock music"], DirectedEdge["Music genre", 
     "Rhythm and blues"], DirectedEdge["Music genre", "Soul music"], 
    DirectedEdge["Music genre", "Reggae"], DirectedEdge["Music genre", 
     "Country"], DirectedEdge["Music genre", "Funk"], 
    DirectedEdge["Music genre", "Folk music"], DirectedEdge["Music genre", 
     "Middle Eastern music"], DirectedEdge["Music genre", "Jazz"], 
    DirectedEdge["Music genre", "Disco"], DirectedEdge["Music genre", 
     "Classical music"], DirectedEdge["Music genre", "Electronic music"], 
    DirectedEdge["Music genre", "Music of Latin America"], 
    DirectedEdge["Music genre", "Blues"], DirectedEdge["Music genre", 
     "Music for children"], DirectedEdge["Music genre", "New-age music"], 
    DirectedEdge["Music genre", "Vocal music"], DirectedEdge["Music genre", 
     "Music of Africa"], DirectedEdge["Music genre", "Christian music"], 
    DirectedEdge["Music genre", "Music of Asia"], DirectedEdge["Music genre", 
     "Ska"], DirectedEdge["Music genre", "Traditional music"], 
    DirectedEdge["Music genre", "Independent music"], 
    DirectedEdge["Hip hop music", "Grime music"], 
    DirectedEdge["Hip hop music", "Trap music"], 
    DirectedEdge["Hip hop music", "Beatboxing"], DirectedEdge["Rock music", 
     "Heavy metal"], DirectedEdge["Rock music", "Punk rock"], 
    DirectedEdge["Rock music", "Grunge"], DirectedEdge["Rock music", 
     "Progressive rock"], DirectedEdge["Rock music", "Rock and roll"], 
    DirectedEdge["Rock music", "Psychedelic rock"], 
    DirectedEdge["Reggae", "Dub"], DirectedEdge["Country", "Swing music"], 
    DirectedEdge["Country", "Bluegrass"], DirectedEdge["Classical music", 
     "Opera"], DirectedEdge["Electronic music", "House music"], 
    DirectedEdge["Electronic music", "Techno"], 
    DirectedEdge["Electronic music", "Dubstep"], 
    DirectedEdge["Electronic music", "Electro"], 
    DirectedEdge["Electronic music", "Drum and bass"], 
    DirectedEdge["Electronic music", "Electronica"], 
    DirectedEdge["Electronic music", "Electronic dance music"], 
    DirectedEdge["Electronic music", "Ambient music"], 
    DirectedEdge["Electronic music", "Trance music"], 
    DirectedEdge["Electronic music", "Noise music"], 
    DirectedEdge["Electronic music", "UK garage"], 
    DirectedEdge["Drum and bass", "Oldschool jungle"], 
    DirectedEdge["Ambient music", "Drone music"], 
    DirectedEdge["Music of Latin America", "Cumbia"], 
    DirectedEdge["Music of Latin America", "Salsa music"], 
    DirectedEdge["Music of Latin America", "Soca music"], 
    DirectedEdge["Music of Latin America", "Kuduro"], 
    DirectedEdge["Music of Latin America", "Funk carioca"], 
    DirectedEdge["Music of Latin America", "Flamenco"], 
    DirectedEdge["Vocal music", "A capella"], DirectedEdge["Vocal music", 
     "Chant"], DirectedEdge["Vocal music", "Beatboxing"], 
    DirectedEdge["Music of Africa", "Afrobeat"], 
    DirectedEdge["Music of Africa", "Kwaito"], 
    DirectedEdge["Christian music", "Gospel music"], 
    DirectedEdge["Music of Asia", "Carnatic music"], 
    DirectedEdge["Music of Asia", "Music of Bollywood"], 
    DirectedEdge["Musical concepts", "Song"], 
    DirectedEdge["Musical concepts", "Melody"], 
    DirectedEdge["Musical concepts", "Musical note"], 
    DirectedEdge["Musical concepts", "Beat"], 
    DirectedEdge["Musical concepts", "Chord"], 
    DirectedEdge["Musical concepts", "Harmony"], 
    DirectedEdge["Musical concepts", "Bassline"], 
    DirectedEdge["Musical concepts", "Loop"], 
    DirectedEdge["Musical concepts", "Drone"], DirectedEdge["Beat", 
     "Drum beat"], DirectedEdge["Music role", "Background music"], 
    DirectedEdge["Music role", "Theme music"], DirectedEdge["Music role", 
     "Jingle (music)"], DirectedEdge["Music role", "Soundtrack music"], 
    DirectedEdge["Music role", "Lullaby"], DirectedEdge["Music role", 
     "Video game music"], DirectedEdge["Music role", "Christmas music"], 
    DirectedEdge["Music role", "Dance music"], DirectedEdge["Music role", 
     "Wedding music"], DirectedEdge["Music role", "Birthday music"], 
    DirectedEdge["Music mood", "Happy music"], DirectedEdge["Music mood", 
     "Funny music"], DirectedEdge["Music mood", "Sad music"], 
    DirectedEdge["Music mood", "Tender music"], DirectedEdge["Music mood", 
     "Exciting music"], DirectedEdge["Music mood", "Angry music"], 
    DirectedEdge["Music mood", "Scary music"], DirectedEdge["Natural sounds", 
     "Wind"], DirectedEdge["Natural sounds", "Thunderstorm"], 
    DirectedEdge["Natural sounds", "Water"], DirectedEdge["Natural sounds", 
     "Fire"], DirectedEdge["Wind", "Howl (wind)"], 
    DirectedEdge["Wind", "Rustling leaves"], DirectedEdge["Wind", 
     "Wind noise (microphone)"], DirectedEdge["Thunderstorm", "Thunder"], 
    DirectedEdge["Water", "Rain"], DirectedEdge["Water", "Stream"], 
    DirectedEdge["Water", "Waterfall"], DirectedEdge["Water", "Ocean"], 
    DirectedEdge["Water", "Steam"], DirectedEdge["Water", "Gurgling"], 
    DirectedEdge["Rain", "Raindrop"], DirectedEdge["Rain", 
     "Rain on surface"], DirectedEdge["Ocean", "Waves, surf"], 
    DirectedEdge["Steam", "Hiss"], DirectedEdge["Fire", "Crackle"], 
    DirectedEdge["Fire", "Wildfire"], DirectedEdge["Sounds of things", 
     "Vehicle"], DirectedEdge["Sounds of things", "Engine"], 
    DirectedEdge["Sounds of things", "Domestic sounds, home sounds"], 
    DirectedEdge["Sounds of things", "Bell"], 
    DirectedEdge["Sounds of things", "Alarm"], 
    DirectedEdge["Sounds of things", "Mechanisms"], 
    DirectedEdge["Sounds of things", "Tools"], 
    DirectedEdge["Sounds of things", "Explosion"], 
    DirectedEdge["Sounds of things", "Wood"], 
    DirectedEdge["Sounds of things", "Glass"], 
    DirectedEdge["Sounds of things", "Liquid"], 
    DirectedEdge["Sounds of things", "Miscellaneous sources"], 
    DirectedEdge["Sounds of things", "Specific impact sounds"], 
    DirectedEdge["Vehicle", "Boat, Water vehicle"], 
    DirectedEdge["Vehicle", "Motor vehicle (road)"], 
    DirectedEdge["Vehicle", "Rail transport"], DirectedEdge["Vehicle", 
     "Aircraft"], DirectedEdge["Vehicle", "Non-motorized land vehicle"], 
    DirectedEdge["Boat, Water vehicle", "Sailboat, sailing ship"], 
    DirectedEdge["Boat, Water vehicle", "Rowboat, canoe, kayak"], 
    DirectedEdge["Boat, Water vehicle", "Motorboat, speedboat"], 
    DirectedEdge["Boat, Water vehicle", "Ship"], 
    DirectedEdge["Motor vehicle (road)", "Car"], 
    DirectedEdge["Motor vehicle (road)", "Truck"], 
    DirectedEdge["Motor vehicle (road)", "Bus"], 
    DirectedEdge["Motor vehicle (road)", "Emergency vehicle"], 
    DirectedEdge["Motor vehicle (road)", "Motorcycle"], 
    DirectedEdge["Motor vehicle (road)", "Traffic noise, roadway noise"], 
    DirectedEdge["Car", "Vehicle horn, car horn, honking"], 
    DirectedEdge["Car", "Car alarm"], DirectedEdge["Car", 
     "Power windows, electric windows"], DirectedEdge["Car", "Skidding"], 
    DirectedEdge["Car", "Tire squeal"], DirectedEdge["Car", 
     "Car passing by"], DirectedEdge["Car", "Race car, auto racing"], 
    DirectedEdge["Vehicle horn, car horn, honking", "Toot"], 
    DirectedEdge["Truck", "Air brake"], DirectedEdge["Truck", 
     "Air horn, truck horn"], DirectedEdge["Truck", "Reversing beeps"], 
    DirectedEdge["Truck", "Ice cream truck, ice cream van"], 
    DirectedEdge["Emergency vehicle", "Police car (siren)"], 
    DirectedEdge["Emergency vehicle", "Ambulance (siren)"], 
    DirectedEdge["Emergency vehicle", "Fire engine, fire truck (siren)"], 
    DirectedEdge["Rail transport", "Train"], DirectedEdge["Rail transport", 
     "Railroad car, train wagon"], DirectedEdge["Rail transport", 
     "Train wheels squealing"], DirectedEdge["Rail transport", 
     "Subway, metro, underground"], DirectedEdge["Train", "Train whistle"], 
    DirectedEdge["Train", "Train horn"], DirectedEdge["Aircraft", 
     "Aircraft engine"], DirectedEdge["Aircraft", "Helicopter"], 
    DirectedEdge["Aircraft", "Fixed-wing aircraft, airplane"], 
    DirectedEdge["Aircraft engine", "Jet engine"], 
    DirectedEdge["Aircraft engine", "Propeller, airscrew"], 
    DirectedEdge["Non-motorized land vehicle", "Bicycle"], 
    DirectedEdge["Non-motorized land vehicle", "Skateboard"], 
    DirectedEdge["Bicycle", "Bicycle bell"], DirectedEdge["Engine", 
     "Light engine (high frequency)"], DirectedEdge["Engine", 
     "Medium engine (mid frequency)"], DirectedEdge["Engine", 
     "Heavy engine (low frequency)"], DirectedEdge["Engine", "Jet engine"], 
    DirectedEdge["Engine", "Engine knocking"], DirectedEdge["Engine", 
     "Engine starting"], DirectedEdge["Engine", "Idling"], 
    DirectedEdge["Engine", "Accelerating, revving, vroom"], 
    DirectedEdge["Light engine (high frequency)", 
     "Dental drill, dentist's drill"], DirectedEdge[
     "Light engine (high frequency)", "Lawn mower"], 
    DirectedEdge["Light engine (high frequency)", "Chainsaw"], 
    DirectedEdge["Domestic sounds, home sounds", "Door"], 
    DirectedEdge["Domestic sounds, home sounds", "Cupboard open or close"], 
    DirectedEdge["Domestic sounds, home sounds", "Drawer open or close"], 
    DirectedEdge["Domestic sounds, home sounds", "Dishes, pots, and pans"], 
    DirectedEdge["Domestic sounds, home sounds", "Cutlery, silverware"], 
    DirectedEdge["Domestic sounds, home sounds", "Chopping (food)"], 
    DirectedEdge["Domestic sounds, home sounds", "Frying (food)"], 
    DirectedEdge["Domestic sounds, home sounds", "Microwave oven"], 
    DirectedEdge["Domestic sounds, home sounds", "Blender"], 
    DirectedEdge["Domestic sounds, home sounds", "Kettle whistle"], 
    DirectedEdge["Domestic sounds, home sounds", "Water tap, faucet"], 
    DirectedEdge["Domestic sounds, home sounds", 
     "Sink (filling or washing)"], DirectedEdge[
     "Domestic sounds, home sounds", "Bathtub (filling or washing)"], 
    DirectedEdge["Domestic sounds, home sounds", "Hair dryer"], 
    DirectedEdge["Domestic sounds, home sounds", "Toilet flush"], 
    DirectedEdge["Domestic sounds, home sounds", "Toothbrush"], 
    DirectedEdge["Domestic sounds, home sounds", "Vacuum cleaner"], 
    DirectedEdge["Domestic sounds, home sounds", "Zipper (clothing)"], 
    DirectedEdge["Domestic sounds, home sounds", 
     "Velcro, hook and loop fastener"], DirectedEdge[
     "Domestic sounds, home sounds", "Keys jangling"], 
    DirectedEdge["Domestic sounds, home sounds", "Coin (dropping)"], 
    DirectedEdge["Domestic sounds, home sounds", "Packing tape, duct tape"], 
    DirectedEdge["Domestic sounds, home sounds", "Scissors"], 
    DirectedEdge["Domestic sounds, home sounds", 
     "Electric shaver, electric razor"], DirectedEdge[
     "Domestic sounds, home sounds", "Shuffling cards"], 
    DirectedEdge["Domestic sounds, home sounds", "Typing"], 
    DirectedEdge["Domestic sounds, home sounds", "Writing"], 
    DirectedEdge["Door", "Doorbell"], DirectedEdge["Door", "Sliding door"], 
    DirectedEdge["Door", "Slam"], DirectedEdge["Door", "Knock"], 
    DirectedEdge["Door", "Tap"], DirectedEdge["Door", "Squeak"], 
    DirectedEdge["Doorbell", "Ding-dong"], DirectedEdge["Toothbrush", 
     "Electric toothbrush"], DirectedEdge["Typing", "Typewriter"], 
    DirectedEdge["Typing", "Computer keyboard"], DirectedEdge["Alarm", 
     "Telephone"], DirectedEdge["Alarm", "Alarm clock"], 
    DirectedEdge["Alarm", "Siren"], DirectedEdge["Alarm", "Doorbell"], 
    DirectedEdge["Alarm", "Buzzer"], DirectedEdge["Alarm", 
     "Smoke detector, smoke alarm"], DirectedEdge["Alarm", "Fire alarm"], 
    DirectedEdge["Alarm", "Car alarm"], DirectedEdge["Alarm", 
     "Vehicle horn, car horn, honking"], DirectedEdge["Alarm", 
     "Bicycle bell"], DirectedEdge["Alarm", "Air horn, truck horn"], 
    DirectedEdge["Alarm", "Foghorn"], DirectedEdge["Alarm", "Whistle"], 
    DirectedEdge["Telephone", "Telephone bell ringing"], 
    DirectedEdge["Telephone", "Ringtone"], DirectedEdge["Telephone", 
     "Cellphone buzz, vibrating alert"], DirectedEdge["Telephone", 
     "Telephone dialing, DTMF"], DirectedEdge["Telephone", "Dial tone"], 
    DirectedEdge["Telephone", "Busy signal"], DirectedEdge["Siren", 
     "Police car (siren)"], DirectedEdge["Siren", "Ambulance (siren)"], 
    DirectedEdge["Siren", "Fire engine, fire truck (siren)"], 
    DirectedEdge["Siren", "Civil defense siren"], DirectedEdge["Whistle", 
     "Kettle whistle"], DirectedEdge["Whistle", "Steam whistle"], 
    DirectedEdge["Mechanisms", "Ratchet, pawl"], DirectedEdge["Mechanisms", 
     "Clock"], DirectedEdge["Mechanisms", "Gears"], 
    DirectedEdge["Mechanisms", "Pulleys"], DirectedEdge["Mechanisms", 
     "Sewing machine"], DirectedEdge["Mechanisms", "Mechanical fan"], 
    DirectedEdge["Mechanisms", "Air conditioning"], 
    DirectedEdge["Mechanisms", "Cash register"], DirectedEdge["Mechanisms", 
     "Printer"], DirectedEdge["Mechanisms", "Camera"], 
    DirectedEdge["Clock", "Tick"], DirectedEdge["Clock", "Tick-tock"], 
    DirectedEdge["Camera", "Single-lens reflex camera"], 
    DirectedEdge["Tools", "Hammer"], DirectedEdge["Tools", "Jackhammer"], 
    DirectedEdge["Tools", "Sawing"], DirectedEdge["Tools", "Filing (rasp)"], 
    DirectedEdge["Tools", "Sanding"], DirectedEdge["Tools", "Power tool"], 
    DirectedEdge["Power tool", "Drill"], DirectedEdge["Drill", 
     "Dental drill, dentist's drill"], DirectedEdge["Explosion", 
     "Gunshot, gunfire"], DirectedEdge["Explosion", "Fireworks"], 
    DirectedEdge["Explosion", "Burst, pop"], DirectedEdge["Explosion", 
     "Eruption"], DirectedEdge["Explosion", "Boom"], 
    DirectedEdge["Gunshot, gunfire", "Machine gun"], 
    DirectedEdge["Gunshot, gunfire", "Fusillade"], 
    DirectedEdge["Gunshot, gunfire", "Artillery fire"], 
    DirectedEdge["Gunshot, gunfire", "Cap gun"], DirectedEdge["Fireworks", 
     "Firecracker"], DirectedEdge["Boom", "Sonic boom"], 
    DirectedEdge["Wood", "Chop"], DirectedEdge["Wood", "Splinter"], 
    DirectedEdge["Wood", "Crack"], DirectedEdge["Wood", "Snap"], 
    DirectedEdge["Glass", "Chink, clink"], DirectedEdge["Glass", "Shatter"], 
    DirectedEdge["Liquid", "Splash, splatter"], DirectedEdge["Liquid", 
     "Squish"], DirectedEdge["Liquid", "Drip"], DirectedEdge["Liquid", 
     "Pour"], DirectedEdge["Liquid", "Fill (with liquid)"], 
    DirectedEdge["Liquid", "Spray"], DirectedEdge["Liquid", "Pump (liquid)"], 
    DirectedEdge["Liquid", "Stir"], DirectedEdge["Liquid", "Boiling"], 
    DirectedEdge["Splash, splatter", "Slosh"], DirectedEdge["Pour", 
     "Trickle, dribble"], DirectedEdge["Pour", "Gush"], 
    DirectedEdge["Miscellaneous sources", "Sonar"], 
    DirectedEdge["Miscellaneous sources", "Duck call (hunting tool)"], 
    DirectedEdge["Miscellaneous sources", "Arrow"], 
    DirectedEdge["Miscellaneous sources", "Sound equipment"], 
    DirectedEdge["Arrow", "Whoosh, swoosh, swish"], 
    DirectedEdge["Arrow", "Thump, thud"], DirectedEdge["Arrow", "Wobble"], 
    DirectedEdge["Thump, thud", "Thunk"], DirectedEdge["Thump, thud", 
     "Clunk"], DirectedEdge["Sound equipment", "Microphone"], 
    DirectedEdge["Sound equipment", "Electronic tuner"], 
    DirectedEdge["Sound equipment", "Guitar amplifier"], 
    DirectedEdge["Sound equipment", "Effects unit"], 
    DirectedEdge["Microphone", "Wind noise (microphone)"], 
    DirectedEdge["Effects unit", "Chorus effect"], 
    DirectedEdge["Specific impact sounds", "Basketball bounce"], 
    DirectedEdge["Source-ambiguous sounds", "Generic impact sounds"], 
    DirectedEdge["Source-ambiguous sounds", "Surface contact"], 
    DirectedEdge["Source-ambiguous sounds", "Deformable shell"], 
    DirectedEdge["Source-ambiguous sounds", "Onomatopoeia"], 
    DirectedEdge["Source-ambiguous sounds", "Silence"], 
    DirectedEdge["Source-ambiguous sounds", "Other sourceless"], 
    DirectedEdge["Generic impact sounds", "Bang"], 
    DirectedEdge["Generic impact sounds", "Slap, smack"], 
    DirectedEdge["Generic impact sounds", "Whack, thwack"], 
    DirectedEdge["Generic impact sounds", "Smash, crash"], 
    DirectedEdge["Generic impact sounds", "Breaking"], 
    DirectedEdge["Generic impact sounds", "Bouncing"], 
    DirectedEdge["Generic impact sounds", "Knock"], 
    DirectedEdge["Generic impact sounds", "Tap"], 
    DirectedEdge["Generic impact sounds", "Thump, thud"], 
    DirectedEdge["Generic impact sounds", "Whip"], 
    DirectedEdge["Generic impact sounds", "Flap"], 
    DirectedEdge["Surface contact", "Scratch"], 
    DirectedEdge["Surface contact", "Scrape"], 
    DirectedEdge["Surface contact", "Rub"], DirectedEdge["Surface contact", 
     "Roll"], DirectedEdge["Surface contact", "Grind"], 
    DirectedEdge["Deformable shell", "Crushing"], 
    DirectedEdge["Deformable shell", "Crumpling, crinkling"], 
    DirectedEdge["Deformable shell", "Tearing"], DirectedEdge["Onomatopoeia", 
     "Brief tone"], DirectedEdge["Onomatopoeia", "Hiss"], 
    DirectedEdge["Onomatopoeia", "Creak"], DirectedEdge["Onomatopoeia", 
     "Rattle"], DirectedEdge["Onomatopoeia", "Whoosh, swoosh, swish"], 
    DirectedEdge["Onomatopoeia", "Rustle"], DirectedEdge["Onomatopoeia", 
     "Whir"], DirectedEdge["Onomatopoeia", "Clatter"], 
    DirectedEdge["Onomatopoeia", "Sizzle"], DirectedEdge["Onomatopoeia", 
     "Clicking"], DirectedEdge["Onomatopoeia", "Rumble"], 
    DirectedEdge["Onomatopoeia", "Blare"], DirectedEdge["Onomatopoeia", 
     "Plop"], DirectedEdge["Onomatopoeia", "Jingle, tinkle"], 
    DirectedEdge["Onomatopoeia", "Fizz"], DirectedEdge["Onomatopoeia", 
     "Puff"], DirectedEdge["Onomatopoeia", "Hum"], 
    DirectedEdge["Onomatopoeia", "Squish"], DirectedEdge["Onomatopoeia", 
     "Zing"], DirectedEdge["Onomatopoeia", "Boing"], 
    DirectedEdge["Onomatopoeia", "Crackle"], DirectedEdge["Onomatopoeia", 
     "Crunch"], DirectedEdge["Onomatopoeia", "Crack"], 
    DirectedEdge["Onomatopoeia", "Snap"], DirectedEdge["Brief tone", 
     "Beep, bleep"], DirectedEdge["Brief tone", "Ping"], 
    DirectedEdge["Brief tone", "Ding"], DirectedEdge["Brief tone", "Clang"], 
    DirectedEdge["Brief tone", "Twang"], DirectedEdge["Brief tone", 
     "Chirp, tweet"], DirectedEdge["Brief tone", "Buzz"], 
    DirectedEdge["Brief tone", "Squeak"], DirectedEdge["Brief tone", 
     "Squeal"], DirectedEdge["Brief tone", "Screech"], 
    DirectedEdge["Clicking", "Tick"], DirectedEdge["Clicking", "Clip-clop"], 
    DirectedEdge["Clicking", "Clickety-clack"], 
    DirectedEdge["Other sourceless", "Sine wave"], 
    DirectedEdge["Other sourceless", "Sound effect"], 
    DirectedEdge["Other sourceless", "Pulse"], 
    DirectedEdge["Other sourceless", "Infrasound"], 
    DirectedEdge["Other sourceless", "Bass (frequency range)"], 
    DirectedEdge["Other sourceless", "Ringing (of resonator)"], 
    DirectedEdge["Sine wave", "Harmonic"], DirectedEdge["Sine wave", 
     "Chirp tone"], DirectedEdge["Channel, environment and background", 
     "Acoustic environment"], DirectedEdge[
     "Channel, environment and background", "Noise"], 
    DirectedEdge["Channel, environment and background", 
     "Sound reproduction"], DirectedEdge["Acoustic environment", 
     "Inside, small room"], DirectedEdge["Acoustic environment", 
     "Inside, large room or hall"], DirectedEdge["Acoustic environment", 
     "Inside, public space"], DirectedEdge["Acoustic environment", 
     "Outside, urban or manmade"], DirectedEdge["Acoustic environment", 
     "Outside, rural or natural"], DirectedEdge["Acoustic environment", 
     "Reverberation"], DirectedEdge["Acoustic environment", "Echo"], 
    DirectedEdge["Noise", "Background noise"], DirectedEdge["Noise", 
     "Hubbub, speech noise, speech babble"], DirectedEdge["Noise", 
     "Cacophony"], DirectedEdge["Noise", "White noise"], 
    DirectedEdge["Noise", "Pink noise"], DirectedEdge["Noise", "Throbbing"], 
    DirectedEdge["Noise", "Vibration"], DirectedEdge["Background noise", 
     "Environmental noise"], DirectedEdge["Background noise", "Tape hiss"], 
    DirectedEdge["Background noise", "Static"], 
    DirectedEdge["Background noise", "Mains hum"], 
    DirectedEdge["Background noise", "Distortion"], 
    DirectedEdge["Background noise", "Sidetone"], 
    DirectedEdge["Sound reproduction", "Television"], 
    DirectedEdge["Sound reproduction", "Radio"], 
    DirectedEdge["Sound reproduction", "Loudspeaker"], 
    DirectedEdge["Sound reproduction", "Headphones"], 
    DirectedEdge["Sound reproduction", "Recording"], 
    DirectedEdge["Sound reproduction", "Gramophone record"], 
    DirectedEdge["Sound reproduction", "Compact disc"], 
    DirectedEdge["Sound reproduction", "MP3"], DirectedEdge["Recording", 
     "Field recording"]}, {GraphLayout -> "RadialEmbedding", 
    Properties -> {"Chord" -> {Tooltip -> "Chord"}, 
      "Flap" -> {Tooltip -> "Flap"}, "Eruption" -> {Tooltip -> "Eruption"}, 
      "Typewriter" -> {Tooltip -> "Typewriter"}, 
      "Mantra" -> {Tooltip -> "Mantra"}, "Pulse" -> {Tooltip -> "Pulse"}, 
      "Mechanisms" -> {Tooltip -> "Mechanisms"}, "Cat communication" -> 
       {Tooltip -> "Cat communication"}, "Bay" -> {Tooltip -> "Bay"}, 
      "Crack" -> {Tooltip -> "Crack"}, "Bagpipes" -> {Tooltip -> "Bagpipes"}, 
      "Growling" -> {Tooltip -> "Growling"}, "Chorus effect" -> 
       {Tooltip -> "Chorus effect"}, "Squeak" -> {Tooltip -> "Squeak"}, 
      "Packing tape, duct tape" -> {Tooltip -> "Packing tape, duct tape"}, 
      "Wind chime" -> {Tooltip -> "Wind chime"}, 
      "Bang" -> {Tooltip -> "Bang"}, "Noise music" -> 
       {Tooltip -> "Noise music"}, "Car" -> {Tooltip -> "Car"}, 
      "Heavy metal" -> {Tooltip -> "Heavy metal"}, 
      "Trap music" -> {Tooltip -> "Trap music"}, 
      "Sanding" -> {Tooltip -> "Sanding"}, "Tuning fork" -> 
       {Tooltip -> "Tuning fork"}, "Biting" -> {Tooltip -> "Biting"}, 
      "Jingle, tinkle" -> {Tooltip -> "Jingle, tinkle"}, 
      "Writing" -> {Tooltip -> "Writing"}, "Jingle (music)" -> 
       {Tooltip -> "Jingle (music)"}, "Smoke detector, smoke alarm" -> 
       {Tooltip -> "Smoke detector, smoke alarm"}, 
      "Yawn" -> {Tooltip -> "Yawn"}, "Soundtrack music" -> 
       {Tooltip -> "Soundtrack music"}, "Angry music" -> 
       {Tooltip -> "Angry music"}, "Vocal music" -> 
       {Tooltip -> "Vocal music"}, "Crackle" -> {Tooltip -> "Crackle"}, 
      "Sliding door" -> {Tooltip -> "Sliding door"}, 
      "Tearing" -> {Tooltip -> "Tearing"}, "Choir" -> {Tooltip -> "Choir"}, 
      "Bee, wasp, etc." -> {Tooltip -> "Bee, wasp, etc."}, 
      "Burst, pop" -> {Tooltip -> "Burst, pop"}, "Explosion" -> 
       {Tooltip -> "Explosion"}, "Melody" -> {Tooltip -> "Melody"}, 
      "Screaming" -> {Tooltip -> "Screaming"}, "Finger snapping" -> 
       {Tooltip -> "Finger snapping"}, "Motorboat, speedboat" -> 
       {Tooltip -> "Motorboat, speedboat"}, "Reggae" -> 
       {Tooltip -> "Reggae"}, "Dog" -> {Tooltip -> "Dog"}, 
      "Sound equipment" -> {Tooltip -> "Sound equipment"}, 
      "Slosh" -> {Tooltip -> "Slosh"}, "Howl" -> {Tooltip -> "Howl"}, 
      "Noise" -> {Tooltip -> "Noise"}, "Alarm" -> {Tooltip -> "Alarm"}, 
      "Horse" -> {Tooltip -> "Horse"}, "Hammer" -> {Tooltip -> "Hammer"}, 
      "Zing" -> {Tooltip -> "Zing"}, "Honk" -> {Tooltip -> "Honk"}, 
      "Brief tone" -> {Tooltip -> "Brief tone"}, 
      "Crow" -> {Tooltip -> "Crow"}, "Door" -> {Tooltip -> "Door"}, 
      "Sheep" -> {Tooltip -> "Sheep"}, "Rhodes piano" -> 
       {Tooltip -> "Rhodes piano"}, "Snare drum" -> 
       {Tooltip -> "Snare drum"}, "Guitar amplifier" -> 
       {Tooltip -> "Guitar amplifier"}, "Sounds of things" -> 
       {Tooltip -> "Sounds of things"}, "Music of Asia" -> 
       {Tooltip -> "Music of Asia"}, "Drone music" -> 
       {Tooltip -> "Drone music"}, "Microphone" -> {Tooltip -> "Microphone"}, 
      "Onomatopoeia" -> {Tooltip -> "Onomatopoeia"}, 
      "Water tap, faucet" -> {Tooltip -> "Water tap, faucet"}, 
      "Oldschool jungle" -> {Tooltip -> "Oldschool jungle"}, 
      "Screech" -> {Tooltip -> "Screech"}, "Engine knocking" -> 
       {Tooltip -> "Engine knocking"}, "Psychedelic rock" -> 
       {Tooltip -> "Psychedelic rock"}, "Motorcycle" -> 
       {Tooltip -> "Motorcycle"}, "Human locomotion" -> 
       {Tooltip -> "Human locomotion"}, "Music of Bollywood" -> 
       {Tooltip -> "Music of Bollywood"}, "Bus" -> {Tooltip -> "Bus"}, 
      "Bluegrass" -> {Tooltip -> "Bluegrass"}, "Kettle whistle" -> 
       {Tooltip -> "Kettle whistle"}, "Double bass" -> 
       {Tooltip -> "Double bass"}, "Bouncing" -> {Tooltip -> "Bouncing"}, 
      "Vehicle horn, car horn, honking" -> 
       {Tooltip -> "Vehicle horn, car horn, honking"}, 
      "Hoot" -> {Tooltip -> "Hoot"}, "Traffic noise, roadway noise" -> 
       {Tooltip -> "Traffic noise, roadway noise"}, 
      "Wood block" -> {Tooltip -> "Wood block"}, 
      "Breaking" -> {Tooltip -> "Breaking"}, "Canidae, dogs, wolves" -> 
       {Tooltip -> "Canidae, dogs, wolves"}, "Crash cymbal" -> 
       {Tooltip -> "Crash cymbal"}, "Progressive rock" -> 
       {Tooltip -> "Progressive rock"}, "Sonic boom" -> 
       {Tooltip -> "Sonic boom"}, "Coo" -> {Tooltip -> "Coo"}, 
      "Miscellaneous sources" -> {Tooltip -> "Miscellaneous sources"}, 
      "Tinnitus, ringing in the ears" -> 
       {Tooltip -> "Tinnitus, ringing in the ears"}, 
      "Independent music" -> {Tooltip -> "Independent music"}, 
      "Emergency vehicle" -> {Tooltip -> "Emergency vehicle"}, 
      "Snap" -> {Tooltip -> "Snap"}, "Cap gun" -> {Tooltip -> "Cap gun"}, 
      "Wind noise (microphone)" -> {Tooltip -> "Wind noise (microphone)"}, 
      "Thunderstorm" -> {Tooltip -> "Thunderstorm"}, 
      "Grind" -> {Tooltip -> "Grind"}, "Foghorn" -> {Tooltip -> "Foghorn"}, 
      "Liquid" -> {Tooltip -> "Liquid"}, "Scrape" -> {Tooltip -> "Scrape"}, 
      "Keyboard (musical)" -> {Tooltip -> "Keyboard (musical)"}, 
      "Jazz" -> {Tooltip -> "Jazz"}, "Chink, clink" -> 
       {Tooltip -> "Chink, clink"}, "Hip hop music" -> 
       {Tooltip -> "Hip hop music"}, "Fly, housefly" -> 
       {Tooltip -> "Fly, housefly"}, "Silence" -> {Tooltip -> "Silence"}, 
      "Rattle" -> {Tooltip -> "Rattle"}, "Cupboard open or close" -> 
       {Tooltip -> "Cupboard open or close"}, 
      "Hi-hat" -> {Tooltip -> "Hi-hat"}, "Disco" -> {Tooltip -> "Disco"}, 
      "Hubbub, speech noise, speech babble" -> 
       {Tooltip -> "Hubbub, speech noise, speech babble"}, 
      "Fizz" -> {Tooltip -> "Fizz"}, "Donkey, ass" -> 
       {Tooltip -> "Donkey, ass"}, "Human sounds" -> 
       {Tooltip -> "Human sounds"}, "Video game music" -> 
       {Tooltip -> "Video game music"}, "Buzz" -> {Tooltip -> "Buzz"}, 
      "Subway, metro, underground" -> 
       {Tooltip -> "Subway, metro, underground"}, "Race car, auto racing" -> 
       {Tooltip -> "Race car, auto racing"}, "Applause" -> 
       {Tooltip -> "Applause"}, "Helicopter" -> {Tooltip -> "Helicopter"}, 
      "Squish" -> {Tooltip -> "Squish"}, "Musical note" -> 
       {Tooltip -> "Musical note"}, "Computer keyboard" -> 
       {Tooltip -> "Computer keyboard"}, "Heavy engine (low frequency)" -> 
       {Tooltip -> "Heavy engine (low frequency)"}, 
      "Boing" -> {Tooltip -> "Boing"}, "Burping, eructation" -> 
       {Tooltip -> "Burping, eructation"}, "Vacuum cleaner" -> 
       {Tooltip -> "Vacuum cleaner"}, "Techno" -> {Tooltip -> "Techno"}, 
      "Organ" -> {Tooltip -> "Organ"}, "Smash, crash" -> 
       {Tooltip -> "Smash, crash"}, "Jet engine" -> 
       {Tooltip -> "Jet engine"}, "Meow" -> {Tooltip -> "Meow"}, 
      "Tap" -> {Tooltip -> "Tap"}, "Sound effect" -> 
       {Tooltip -> "Sound effect"}, "Creak" -> {Tooltip -> "Creak"}, 
      "Tender music" -> {Tooltip -> "Tender music"}, 
      "Goose" -> {Tooltip -> "Goose"}, "Raindrop" -> {Tooltip -> "Raindrop"}, 
      "Machine gun" -> {Tooltip -> "Machine gun"}, "Conversation" -> 
       {Tooltip -> "Conversation"}, "Microwave oven" -> 
       {Tooltip -> "Microwave oven"}, "Keys jangling" -> 
       {Tooltip -> "Keys jangling"}, "Crunch" -> {Tooltip -> "Crunch"}, 
      "Drill" -> {Tooltip -> "Drill"}, "Church bell" -> 
       {Tooltip -> "Church bell"}, "Rail transport" -> 
       {Tooltip -> "Rail transport"}, "Electric shaver, electric razor" -> 
       {Tooltip -> "Electric shaver, electric razor"}, 
      "Snort (horse)" -> {Tooltip -> "Snort (horse)"}, 
      "Truck" -> {Tooltip -> "Truck"}, "Wobble" -> {Tooltip -> "Wobble"}, 
      "Whimper (dog)" -> {Tooltip -> "Whimper (dog)"}, 
      "Coin (dropping)" -> {Tooltip -> "Coin (dropping)"}, 
      "Rain" -> {Tooltip -> "Rain"}, "Dental drill, dentist's drill" -> 
       {Tooltip -> "Dental drill, dentist's drill"}, 
      "Vibration" -> {Tooltip -> "Vibration"}, "Crying, sobbing" -> 
       {Tooltip -> "Crying, sobbing"}, "Railroad car, train wagon" -> 
       {Tooltip -> "Railroad car, train wagon"}, "Tick-tock" -> 
       {Tooltip -> "Tick-tock"}, "Rub" -> {Tooltip -> "Rub"}, 
      "Soca music" -> {Tooltip -> "Soca music"}, "Marimba, xylophone" -> 
       {Tooltip -> "Marimba, xylophone"}, "Squawk" -> {Tooltip -> "Squawk"}, 
      "Fire engine, fire truck (siren)" -> 
       {Tooltip -> "Fire engine, fire truck (siren)"}, 
      "Domestic sounds, home sounds" -> 
       {Tooltip -> "Domestic sounds, home sounds"}, "Gramophone record" -> 
       {Tooltip -> "Gramophone record"}, "Clock" -> {Tooltip -> "Clock"}, 
      "Thump, thud" -> {Tooltip -> "Thump, thud"}, 
      "Grunge" -> {Tooltip -> "Grunge"}, "Electric piano" -> 
       {Tooltip -> "Electric piano"}, "Drum beat" -> 
       {Tooltip -> "Drum beat"}, "Bowed string instrument" -> 
       {Tooltip -> "Bowed string instrument"}, "Drum kit" -> 
       {Tooltip -> "Drum kit"}, "Alarm clock" -> {Tooltip -> "Alarm clock"}, 
      "Arrow" -> {Tooltip -> "Arrow"}, "Punk rock" -> 
       {Tooltip -> "Punk rock"}, "Gasp" -> {Tooltip -> "Gasp"}, 
      "Drum roll" -> {Tooltip -> "Drum roll"}, "Saxophone" -> 
       {Tooltip -> "Saxophone"}, "Singing bowl" -> 
       {Tooltip -> "Singing bowl"}, "Clang" -> {Tooltip -> "Clang"}, 
      "Opera" -> {Tooltip -> "Opera"}, "Chicken, rooster" -> 
       {Tooltip -> "Chicken, rooster"}, "Electronic dance music" -> 
       {Tooltip -> "Electronic dance music"}, 
      "Rumble" -> {Tooltip -> "Rumble"}, "Fill (with liquid)" -> 
       {Tooltip -> "Fill (with liquid)"}, "Music of Latin America" -> 
       {Tooltip -> "Music of Latin America"}, "Pink noise" -> 
       {Tooltip -> "Pink noise"}, "Classical music" -> 
       {Tooltip -> "Classical music"}, "Fire" -> {Tooltip -> "Fire"}, 
      "Tapping (guitar technique)" -> 
       {Tooltip -> "Tapping (guitar technique)"}, 
      "Caterwaul" -> {Tooltip -> "Caterwaul"}, "Effects unit" -> 
       {Tooltip -> "Effects unit"}, "Ping" -> {Tooltip -> "Ping"}, 
      "Grunt" -> {Tooltip -> "Grunt"}, "Dubstep" -> {Tooltip -> "Dubstep"}, 
      "Headphones" -> {Tooltip -> "Headphones"}, "Train whistle" -> 
       {Tooltip -> "Train whistle"}, "Wedding music" -> 
       {Tooltip -> "Wedding music"}, "Throat clearing" -> 
       {Tooltip -> "Throat clearing"}, "Wild animals" -> 
       {Tooltip -> "Wild animals"}, "Sniff" -> {Tooltip -> "Sniff"}, 
      "Telephone bell ringing" -> {Tooltip -> "Telephone bell ringing"}, 
      "Cutlery, silverware" -> {Tooltip -> "Cutlery, silverware"}, 
      "Rain on surface" -> {Tooltip -> "Rain on surface"}, 
      "Bicycle" -> {Tooltip -> "Bicycle"}, "Scary music" -> 
       {Tooltip -> "Scary music"}, "Piano" -> {Tooltip -> "Piano"}, 
      "Snoring" -> {Tooltip -> "Snoring"}, "Cat" -> {Tooltip -> "Cat"}, 
      "Fireworks" -> {Tooltip -> "Fireworks"}, "Whir" -> {Tooltip -> "Whir"}, 
      "Loop" -> {Tooltip -> "Loop"}, "Kuduro" -> {Tooltip -> "Kuduro"}, 
      "Neigh, whinny" -> {Tooltip -> "Neigh, whinny"}, 
      "Sigh" -> {Tooltip -> "Sigh"}, "Croak" -> {Tooltip -> "Croak"}, 
      "Mouse" -> {Tooltip -> "Mouse"}, "Puff" -> {Tooltip -> "Puff"}, 
      "Lawn mower" -> {Tooltip -> "Lawn mower"}, 
      "Glass" -> {Tooltip -> "Glass"}, "Yip" -> {Tooltip -> "Yip"}, 
      "Whoosh, swoosh, swish" -> {Tooltip -> "Whoosh, swoosh, swish"}, 
      "Trumpet" -> {Tooltip -> "Trumpet"}, "Pump (liquid)" -> 
       {Tooltip -> "Pump (liquid)"}, "Stomach rumble" -> 
       {Tooltip -> "Stomach rumble"}, "Theme music" -> 
       {Tooltip -> "Theme music"}, "Dub" -> {Tooltip -> "Dub"}, 
      "Heart sounds, heartbeat" -> {Tooltip -> "Heart sounds, heartbeat"}, 
      "Baby laughter" -> {Tooltip -> "Baby laughter"}, 
      "Tire squeal" -> {Tooltip -> "Tire squeal"}, 
      "Train wheels squealing" -> {Tooltip -> "Train wheels squealing"}, 
      "Shout" -> {Tooltip -> "Shout"}, "MP3" -> {Tooltip -> "MP3"}, 
      "Skateboard" -> {Tooltip -> "Skateboard"}, 
      "Mandolin" -> {Tooltip -> "Mandolin"}, "Gargling" -> 
       {Tooltip -> "Gargling"}, "Wildfire" -> {Tooltip -> "Wildfire"}, 
      "Specific impact sounds" -> {Tooltip -> "Specific impact sounds"}, 
      "UK garage" -> {Tooltip -> "UK garage"}, "Sine wave" -> 
       {Tooltip -> "Sine wave"}, "Bass guitar" -> {Tooltip -> "Bass guitar"}, 
      "Sink (filling or washing)" -> 
       {Tooltip -> "Sink (filling or washing)"}, "Belly laugh" -> 
       {Tooltip -> "Belly laugh"}, "Propeller, airscrew" -> 
       {Tooltip -> "Propeller, airscrew"}, "Blender" -> 
       {Tooltip -> "Blender"}, "Sewing machine" -> 
       {Tooltip -> "Sewing machine"}, "Wail, moan" -> 
       {Tooltip -> "Wail, moan"}, "Compact disc" -> 
       {Tooltip -> "Compact disc"}, "Whale vocalization" -> 
       {Tooltip -> "Whale vocalization"}, "French horn" -> 
       {Tooltip -> "French horn"}, "Bassline" -> {Tooltip -> "Bassline"}, 
      "Shatter" -> {Tooltip -> "Shatter"}, "Whistle" -> 
       {Tooltip -> "Whistle"}, "Idling" -> {Tooltip -> "Idling"}, 
      "Cluck" -> {Tooltip -> "Cluck"}, "Inside, small room" -> 
       {Tooltip -> "Inside, small room"}, "Drum" -> {Tooltip -> "Drum"}, 
      "Pulleys" -> {Tooltip -> "Pulleys"}, "Clunk" -> {Tooltip -> "Clunk"}, 
      "Ska" -> {Tooltip -> "Ska"}, "Car alarm" -> {Tooltip -> "Car alarm"}, 
      "Vehicle" -> {Tooltip -> "Vehicle"}, "Bassoon" -> 
       {Tooltip -> "Bassoon"}, "Zipper (clothing)" -> 
       {Tooltip -> "Zipper (clothing)"}, "Radio" -> {Tooltip -> "Radio"}, 
      "Percussion" -> {Tooltip -> "Percussion"}, 
      "Lullaby" -> {Tooltip -> "Lullaby"}, "Loudspeaker" -> 
       {Tooltip -> "Loudspeaker"}, "Fart" -> {Tooltip -> "Fart"}, 
      "Yell" -> {Tooltip -> "Yell"}, "Drum and bass" -> 
       {Tooltip -> "Drum and bass"}, "Bell" -> {Tooltip -> "Bell"}, 
      "Synthesizer" -> {Tooltip -> "Synthesizer"}, 
      "Dishes, pots, and pans" -> {Tooltip -> "Dishes, pots, and pans"}, 
      "Hair dryer" -> {Tooltip -> "Hair dryer"}, "Digestive" -> 
       {Tooltip -> "Digestive"}, "Gunshot, gunfire" -> 
       {Tooltip -> "Gunshot, gunfire"}, "Brass instrument" -> 
       {Tooltip -> "Brass instrument"}, 
      "Scratching (performance technique)" -> 
       {Tooltip -> "Scratching (performance technique)"}, 
      "Chipmunk" -> {Tooltip -> "Chipmunk"}, "Clickety-clack" -> 
       {Tooltip -> "Clickety-clack"}, "Timpani" -> {Tooltip -> "Timpani"}, 
      "Gong" -> {Tooltip -> "Gong"}, "Ratchet, pawl" -> 
       {Tooltip -> "Ratchet, pawl"}, "Velcro, hook and loop fastener" -> 
       {Tooltip -> "Velcro, hook and loop fastener"}, 
      "Singing" -> {Tooltip -> "Singing"}, "Roar" -> {Tooltip -> "Roar"}, 
      "Shofar" -> {Tooltip -> "Shofar"}, "Pigeon, dove" -> 
       {Tooltip -> "Pigeon, dove"}, "Ambient music" -> 
       {Tooltip -> "Ambient music"}, "Pop music" -> {Tooltip -> "Pop music"}, 
      "Background noise" -> {Tooltip -> "Background noise"}, 
      "Plucked string instrument" -> 
       {Tooltip -> "Plucked string instrument"}, "Sad music" -> 
       {Tooltip -> "Sad music"}, "Yak" -> {Tooltip -> "Yak"}, 
      "Tambourine" -> {Tooltip -> "Tambourine"}, 
      "Stir" -> {Tooltip -> "Stir"}, "Baby cry, infant cry" -> 
       {Tooltip -> "Baby cry, infant cry"}, "Male singing" -> 
       {Tooltip -> "Male singing"}, "Boiling" -> {Tooltip -> "Boiling"}, 
      "Traditional music" -> {Tooltip -> "Traditional music"}, 
      "Electronica" -> {Tooltip -> "Electronica"}, 
      "Siren" -> {Tooltip -> "Siren"}, "Flamenco" -> {Tooltip -> "Flamenco"}, 
      "Electric guitar" -> {Tooltip -> "Electric guitar"}, 
      "Accordion" -> {Tooltip -> "Accordion"}, "Pant" -> {Tooltip -> "Pant"}, 
      "Electronic tuner" -> {Tooltip -> "Electronic tuner"}, 
      "Steam whistle" -> {Tooltip -> "Steam whistle"}, 
      "Frog" -> {Tooltip -> "Frog"}, "Water" -> {Tooltip -> "Water"}, 
      "Funk" -> {Tooltip -> "Funk"}, "Child speech, kid speaking" -> 
       {Tooltip -> "Child speech, kid speaking"}, 
      "Clapping" -> {Tooltip -> "Clapping"}, "Whoop" -> {Tooltip -> "Whoop"}, 
      "Engine" -> {Tooltip -> "Engine"}, "Zither" -> {Tooltip -> "Zither"}, 
      "Mosquito" -> {Tooltip -> "Mosquito"}, 
      "Fixed-wing aircraft, airplane" -> 
       {Tooltip -> "Fixed-wing aircraft, airplane"}, 
      "Funk carioca" -> {Tooltip -> "Funk carioca"}, 
      "Static" -> {Tooltip -> "Static"}, "Trance music" -> 
       {Tooltip -> "Trance music"}, "Music of Africa" -> 
       {Tooltip -> "Music of Africa"}, "Acoustic guitar" -> 
       {Tooltip -> "Acoustic guitar"}, "Hammond organ" -> 
       {Tooltip -> "Hammond organ"}, "Human voice" -> 
       {Tooltip -> "Human voice"}, "Banjo" -> {Tooltip -> "Banjo"}, 
      "Medium engine (mid frequency)" -> 
       {Tooltip -> "Medium engine (mid frequency)"}, 
      "Natural sounds" -> {Tooltip -> "Natural sounds"}, 
      "Gull, seagull" -> {Tooltip -> "Gull, seagull"}, 
      "Cellphone buzz, vibrating alert" -> 
       {Tooltip -> "Cellphone buzz, vibrating alert"}, 
      "Howl (wind)" -> {Tooltip -> "Howl (wind)"}, "Heart murmur" -> 
       {Tooltip -> "Heart murmur"}, "Surface contact" -> 
       {Tooltip -> "Surface contact"}, "Bellow" -> {Tooltip -> "Bellow"}, 
      "Roaring cats (lions, tigers)" -> 
       {Tooltip -> "Roaring cats (lions, tigers)"}, 
      "Steam" -> {Tooltip -> "Steam"}, "Female singing" -> 
       {Tooltip -> "Female singing"}, "Snicker" -> {Tooltip -> "Snicker"}, 
      "Chewing, mastication" -> {Tooltip -> "Chewing, mastication"}, 
      "Respiratory sounds" -> {Tooltip -> "Respiratory sounds"}, 
      "Sound reproduction" -> {Tooltip -> "Sound reproduction"}, 
      "Other sourceless" -> {Tooltip -> "Other sourceless"}, 
      "Birthday music" -> {Tooltip -> "Birthday music"}, 
      "Swing music" -> {Tooltip -> "Swing music"}, 
      "Splinter" -> {Tooltip -> "Splinter"}, "Hiccup" -> 
       {Tooltip -> "Hiccup"}, "House music" -> {Tooltip -> "House music"}, 
      "Speech" -> {Tooltip -> "Speech"}, "Trombone" -> 
       {Tooltip -> "Trombone"}, "Rustle" -> {Tooltip -> "Rustle"}, 
      "Shuffle" -> {Tooltip -> "Shuffle"}, "Sizzle" -> {Tooltip -> "Sizzle"}, 
      "Ringing (of resonator)" -> {Tooltip -> "Ringing (of resonator)"}, 
      "Slam" -> {Tooltip -> "Slam"}, "Spray" -> {Tooltip -> "Spray"}, 
      "Mellotron" -> {Tooltip -> "Mellotron"}, 
      "Ocean" -> {Tooltip -> "Ocean"}, "Inside, public space" -> 
       {Tooltip -> "Inside, public space"}, "Trickle, dribble" -> 
       {Tooltip -> "Trickle, dribble"}, "Television" -> 
       {Tooltip -> "Television"}, "Vibraphone" -> {Tooltip -> "Vibraphone"}, 
      "Crowd" -> {Tooltip -> "Crowd"}, "Ship" -> {Tooltip -> "Ship"}, 
      "Wood" -> {Tooltip -> "Wood"}, "Whimper" -> {Tooltip -> "Whimper"}, 
      "Drip" -> {Tooltip -> "Drip"}, "Ambulance (siren)" -> 
       {Tooltip -> "Ambulance (siren)"}, "Hum" -> {Tooltip -> "Hum"}, 
      "Cattle, bovinae" -> {Tooltip -> "Cattle, bovinae"}, 
      "Camera" -> {Tooltip -> "Camera"}, "Soul music" -> 
       {Tooltip -> "Soul music"}, "Source-ambiguous sounds" -> 
       {Tooltip -> "Source-ambiguous sounds"}, "Clicking" -> 
       {Tooltip -> "Clicking"}, "Grime music" -> {Tooltip -> "Grime music"}, 
      "Soprano saxophone" -> {Tooltip -> "Soprano saxophone"}, 
      "Sampler" -> {Tooltip -> "Sampler"}, "Bugle" -> {Tooltip -> "Bugle"}, 
      "Tools" -> {Tooltip -> "Tools"}, "Air brake" -> 
       {Tooltip -> "Air brake"}, "Air horn, truck horn" -> 
       {Tooltip -> "Air horn, truck horn"}, "Moo" -> {Tooltip -> "Moo"}, 
      "Buzzer" -> {Tooltip -> "Buzzer"}, "Typing" -> {Tooltip -> "Typing"}, 
      "Throbbing" -> {Tooltip -> "Throbbing"}, "Tick" -> {Tooltip -> "Tick"}, 
      "Nicker" -> {Tooltip -> "Nicker"}, "Recording" -> 
       {Tooltip -> "Recording"}, "Electric toothbrush" -> 
       {Tooltip -> "Electric toothbrush"}, "Female speech, woman speaking" -> 
       {Tooltip -> "Female speech, woman speaking"}, 
      "Otoacoustic emission" -> {Tooltip -> "Otoacoustic emission"}, 
      "Rapping" -> {Tooltip -> "Rapping"}, "Exciting music" -> 
       {Tooltip -> "Exciting music"}, "Caw" -> {Tooltip -> "Caw"}, 
      "Cowbell" -> {Tooltip -> "Cowbell"}, "Rimshot" -> 
       {Tooltip -> "Rimshot"}, "Strum" -> {Tooltip -> "Strum"}, 
      "Gears" -> {Tooltip -> "Gears"}, "Duck" -> {Tooltip -> "Duck"}, 
      "Bow-wow" -> {Tooltip -> "Bow-wow"}, "A capella" -> 
       {Tooltip -> "A capella"}, "Train" -> {Tooltip -> "Train"}, 
      "Chirp, tweet" -> {Tooltip -> "Chirp, tweet"}, 
      "Cymbal" -> {Tooltip -> "Cymbal"}, "Pour" -> {Tooltip -> "Pour"}, 
      "Basketball bounce" -> {Tooltip -> "Basketball bounce"}, 
      "Male speech, man speaking" -> 
       {Tooltip -> "Male speech, man speaking"}, 
      "Toot" -> {Tooltip -> "Toot"}, "Acoustic environment" -> 
       {Tooltip -> "Acoustic environment"}, "Guitar" -> 
       {Tooltip -> "Guitar"}, "Chirp tone" -> {Tooltip -> "Chirp tone"}, 
      "Change ringing (campanology)" -> 
       {Tooltip -> "Change ringing (campanology)"}, "Crumpling, crinkling" -> 
       {Tooltip -> "Crumpling, crinkling"}, "Scratch" -> 
       {Tooltip -> "Scratch"}, "Ding-dong" -> {Tooltip -> "Ding-dong"}, 
      "Blare" -> {Tooltip -> "Blare"}, "Rustling leaves" -> 
       {Tooltip -> "Rustling leaves"}, "Pig" -> {Tooltip -> "Pig"}, 
      "Animal" -> {Tooltip -> "Animal"}, "Country" -> {Tooltip -> "Country"}, 
      "Non-motorized land vehicle" -> 
       {Tooltip -> "Non-motorized land vehicle"}, 
      "Boom" -> {Tooltip -> "Boom"}, "String section" -> 
       {Tooltip -> "String section"}, "Deformable shell" -> 
       {Tooltip -> "Deformable shell"}, "Car passing by" -> 
       {Tooltip -> "Car passing by"}, "Artillery fire" -> 
       {Tooltip -> "Artillery fire"}, "Narration, monologue" -> 
       {Tooltip -> "Narration, monologue"}, "Snort" -> {Tooltip -> "Snort"}, 
      "Chop" -> {Tooltip -> "Chop"}, "Babbling" -> {Tooltip -> "Babbling"}, 
      "Gurgling" -> {Tooltip -> "Gurgling"}, 
      "Power windows, electric windows" -> 
       {Tooltip -> "Power windows, electric windows"}, 
      "Tubular bells" -> {Tooltip -> "Tubular bells"}, 
      "Livestock, farm animals, working animals" -> 
       {Tooltip -> "Livestock, farm animals, working animals"}, 
      "Fire alarm" -> {Tooltip -> "Fire alarm"}, "Harpsichord" -> 
       {Tooltip -> "Harpsichord"}, "Harmonic" -> {Tooltip -> "Harmonic"}, 
      "Bass (instrument role)" -> {Tooltip -> "Bass (instrument role)"}, 
      "Carnatic music" -> {Tooltip -> "Carnatic music"}, 
      "Groan" -> {Tooltip -> "Groan"}, "Accelerating, revving, vroom" -> 
       {Tooltip -> "Accelerating, revving, vroom"}, "Violin, fiddle" -> 
       {Tooltip -> "Violin, fiddle"}, 
      "Channel, environment and background" -> 
       {Tooltip -> "Channel, environment and background"}, 
      "Waterfall" -> {Tooltip -> "Waterfall"}, "Harmonica" -> 
       {Tooltip -> "Harmonica"}, "Flute" -> {Tooltip -> "Flute"}, 
      "Rhythm and blues" -> {Tooltip -> "Rhythm and blues"}, 
      "Chuckle, chortle" -> {Tooltip -> "Chuckle, chortle"}, 
      "Drone" -> {Tooltip -> "Drone"}, "Chopping (food)" -> 
       {Tooltip -> "Chopping (food)"}, "Cacophony" -> 
       {Tooltip -> "Cacophony"}, "Clatter" -> {Tooltip -> "Clatter"}, 
      "Musical ensemble" -> {Tooltip -> "Musical ensemble"}, 
      "Ding" -> {Tooltip -> "Ding"}, "Jackhammer" -> 
       {Tooltip -> "Jackhammer"}, "Wolf-whistling" -> 
       {Tooltip -> "Wolf-whistling"}, "Walk, footsteps" -> 
       {Tooltip -> "Walk, footsteps"}, "Tabla" -> {Tooltip -> "Tabla"}, 
      "Beatboxing" -> {Tooltip -> "Beatboxing"}, "Drawer open or close" -> 
       {Tooltip -> "Drawer open or close"}, "Ukulele" -> 
       {Tooltip -> "Ukulele"}, "Steel guitar, slide guitar" -> 
       {Tooltip -> "Steel guitar, slide guitar"}, 
      "Crowing, cock-a-doodle-doo" -> 
       {Tooltip -> "Crowing, cock-a-doodle-doo"}, 
      "Laughter" -> {Tooltip -> "Laughter"}, "Snake" -> {Tooltip -> "Snake"}, 
      "Civil defense siren" -> {Tooltip -> "Civil defense siren"}, 
      "Bird vocalization, bird call, bird song" -> 
       {Tooltip -> "Bird vocalization, bird call, bird song"}, 
      "Giggle" -> {Tooltip -> "Giggle"}, "Mallet percussion" -> 
       {Tooltip -> "Mallet percussion"}, "Ice cream truck, ice cream van" -> 
       {Tooltip -> "Ice cream truck, ice cream van"}, 
      "Harp" -> {Tooltip -> "Harp"}, "Run" -> {Tooltip -> "Run"}, 
      "Wind instrument, woodwind instrument" -> 
       {Tooltip -> "Wind instrument, woodwind instrument"}, 
      "Rock music" -> {Tooltip -> "Rock music"}, 
      "Blues" -> {Tooltip -> "Blues"}, "Cumbia" -> {Tooltip -> "Cumbia"}, 
      "Booing" -> {Tooltip -> "Booing"}, "Thunder" -> {Tooltip -> "Thunder"}, 
      "Music role" -> {Tooltip -> "Music role"}, "Music genre" -> 
       {Tooltip -> "Music genre"}, "Breathing" -> {Tooltip -> "Breathing"}, 
      "Chatter" -> {Tooltip -> "Chatter"}, "Electronic organ" -> 
       {Tooltip -> "Electronic organ"}, "Chainsaw" -> 
       {Tooltip -> "Chainsaw"}, "Cello" -> {Tooltip -> "Cello"}, 
      "Human group actions" -> {Tooltip -> "Human group actions"}, 
      "New-age music" -> {Tooltip -> "New-age music"}, 
      "Whip" -> {Tooltip -> "Whip"}, "Chant" -> {Tooltip -> "Chant"}, 
      "Fusillade" -> {Tooltip -> "Fusillade"}, "Echo" -> {Tooltip -> "Echo"}, 
      "Insect" -> {Tooltip -> "Insect"}, "Firecracker" -> 
       {Tooltip -> "Firecracker"}, "Electro" -> {Tooltip -> "Electro"}, 
      "Cheering" -> {Tooltip -> "Cheering"}, "Background music" -> 
       {Tooltip -> "Background music"}, "Outside, rural or natural" -> 
       {Tooltip -> "Outside, rural or natural"}, "Christian music" -> 
       {Tooltip -> "Christian music"}, "Sneeze" -> {Tooltip -> "Sneeze"}, 
      "Rowboat, canoe, kayak" -> {Tooltip -> "Rowboat, canoe, kayak"}, 
      "Sonar" -> {Tooltip -> "Sonar"}, "Afrobeat" -> {Tooltip -> "Afrobeat"}, 
      "Dance music" -> {Tooltip -> "Dance music"}, 
      "Sailboat, sailing ship" -> {Tooltip -> "Sailboat, sailing ship"}, 
      "Children shouting" -> {Tooltip -> "Children shouting"}, 
      "Bass drum" -> {Tooltip -> "Bass drum"}, "Whack, thwack" -> 
       {Tooltip -> "Whack, thwack"}, "Stream" -> {Tooltip -> "Stream"}, 
      "Rock and roll" -> {Tooltip -> "Rock and roll"}, 
      "Clip-clop" -> {Tooltip -> "Clip-clop"}, 
      "Knock" -> {Tooltip -> "Knock"}, "Music for children" -> 
       {Tooltip -> "Music for children"}, "Purr" -> {Tooltip -> "Purr"}, 
      "Oink" -> {Tooltip -> "Oink"}, "Bicycle bell" -> 
       {Tooltip -> "Bicycle bell"}, "Jingle bell" -> 
       {Tooltip -> "Jingle bell"}, "Ringtone" -> {Tooltip -> "Ringtone"}, 
      "Turkey" -> {Tooltip -> "Turkey"}, "Kwaito" -> {Tooltip -> "Kwaito"}, 
      "Alto saxophone" -> {Tooltip -> "Alto saxophone"}, 
      "Musical concepts" -> {Tooltip -> "Musical concepts"}, 
      "Glockenspiel" -> {Tooltip -> "Glockenspiel"}, 
      "Goat" -> {Tooltip -> "Goat"}, "Patter" -> {Tooltip -> "Patter"}, 
      "Theremin" -> {Tooltip -> "Theremin"}, "Didgeridoo" -> 
       {Tooltip -> "Didgeridoo"}, "Beat" -> {Tooltip -> "Beat"}, 
      "Song" -> {Tooltip -> "Song"}, "Busy signal" -> 
       {Tooltip -> "Busy signal"}, "Pizzicato" -> {Tooltip -> "Pizzicato"}, 
      "Motor vehicle (road)" -> {Tooltip -> "Motor vehicle (road)"}, 
      "Musical instrument" -> {Tooltip -> "Musical instrument"}, 
      "Mains hum" -> {Tooltip -> "Mains hum"}, "Gush" -> {Tooltip -> "Gush"}, 
      "Environmental noise" -> {Tooltip -> "Environmental noise"}, 
      "Scissors" -> {Tooltip -> "Scissors"}, "Roll" -> {Tooltip -> "Roll"}, 
      "Toothbrush" -> {Tooltip -> "Toothbrush"}, "Music mood" -> 
       {Tooltip -> "Music mood"}, "Speech synthesizer" -> 
       {Tooltip -> "Speech synthesizer"}, "Beep, bleep" -> 
       {Tooltip -> "Beep, bleep"}, "Telephone dialing, DTMF" -> 
       {Tooltip -> "Telephone dialing, DTMF"}, "Christmas music" -> 
       {Tooltip -> "Christmas music"}, "Air conditioning" -> 
       {Tooltip -> "Air conditioning"}, "Quack" -> {Tooltip -> "Quack"}, 
      "Bark" -> {Tooltip -> "Bark"}, "Inside, large room or hall" -> 
       {Tooltip -> "Inside, large room or hall"}, 
      "Wind" -> {Tooltip -> "Wind"}, "Light engine (high frequency)" -> 
       {Tooltip -> "Light engine (high frequency)"}, 
      "Power tool" -> {Tooltip -> "Power tool"}, 
      "Gobble" -> {Tooltip -> "Gobble"}, "Twang" -> {Tooltip -> "Twang"}, 
      "Maraca" -> {Tooltip -> "Maraca"}, "Funny music" -> 
       {Tooltip -> "Funny music"}, "Aircraft engine" -> 
       {Tooltip -> "Aircraft engine"}, "Engine starting" -> 
       {Tooltip -> "Engine starting"}, "Bleat" -> {Tooltip -> "Bleat"}, 
      "Wheeze" -> {Tooltip -> "Wheeze"}, "Skidding" -> 
       {Tooltip -> "Skidding"}, "Splash, splatter" -> 
       {Tooltip -> "Splash, splatter"}, "Child singing" -> 
       {Tooltip -> "Child singing"}, "Police car (siren)" -> 
       {Tooltip -> "Police car (siren)"}, "Domestic animals, pets" -> 
       {Tooltip -> "Domestic animals, pets"}, "Drum machine" -> 
       {Tooltip -> "Drum machine"}, "Humming" -> {Tooltip -> "Humming"}, 
      "Duck call (hunting tool)" -> {Tooltip -> "Duck call (hunting tool)"}, 
      "Single-lens reflex camera" -> 
       {Tooltip -> "Single-lens reflex camera"}, "Gospel music" -> 
       {Tooltip -> "Gospel music"}, "Whistling" -> {Tooltip -> "Whistling"}, 
      "Mechanical fan" -> {Tooltip -> "Mechanical fan"}, 
      "Rodents, rats, mice" -> {Tooltip -> "Rodents, rats, mice"}, 
      "Bass (frequency range)" -> {Tooltip -> "Bass (frequency range)"}, 
      "Whispering" -> {Tooltip -> "Whispering"}, 
      "Bird flight, flapping wings" -> 
       {Tooltip -> "Bird flight, flapping wings"}, 
      "Oboe" -> {Tooltip -> "Oboe"}, "Fowl" -> {Tooltip -> "Fowl"}, 
      "Synthetic singing" -> {Tooltip -> "Synthetic singing"}, 
      "Steelpan" -> {Tooltip -> "Steelpan"}, "Tape hiss" -> 
       {Tooltip -> "Tape hiss"}, "Reverberation" -> 
       {Tooltip -> "Reverberation"}, "Sawing" -> {Tooltip -> "Sawing"}, 
      "Telephone" -> {Tooltip -> "Telephone"}, "Salsa music" -> 
       {Tooltip -> "Salsa music"}, "Folk music" -> {Tooltip -> "Folk music"}, 
      "Bathtub (filling or washing)" -> 
       {Tooltip -> "Bathtub (filling or washing)"}, "Middle Eastern music" -> 
       {Tooltip -> "Middle Eastern music"}, "Generic impact sounds" -> 
       {Tooltip -> "Generic impact sounds"}, "Battle cry" -> 
       {Tooltip -> "Battle cry"}, "Yodeling" -> {Tooltip -> "Yodeling"}, 
      "Doorbell" -> {Tooltip -> "Doorbell"}, "Bird" -> {Tooltip -> "Bird"}, 
      "Aircraft" -> {Tooltip -> "Aircraft"}, "Harmony" -> 
       {Tooltip -> "Harmony"}, "Boat, Water vehicle" -> 
       {Tooltip -> "Boat, Water vehicle"}, "Electronic music" -> 
       {Tooltip -> "Electronic music"}, "Printer" -> {Tooltip -> "Printer"}, 
      "Infrasound" -> {Tooltip -> "Infrasound"}, 
      "Crushing" -> {Tooltip -> "Crushing"}, "Clavinet" -> 
       {Tooltip -> "Clavinet"}, "Outside, urban or manmade" -> 
       {Tooltip -> "Outside, urban or manmade"}, 
      "Music" -> {Tooltip -> "Music"}, "Happy music" -> 
       {Tooltip -> "Happy music"}, "Filing (rasp)" -> 
       {Tooltip -> "Filing (rasp)"}, "Squeal" -> {Tooltip -> "Squeal"}, 
      "Children playing" -> {Tooltip -> "Children playing"}, 
      "Cricket" -> {Tooltip -> "Cricket"}, "White noise" -> 
       {Tooltip -> "White noise"}, "Sitar" -> {Tooltip -> "Sitar"}, 
      "Thunk" -> {Tooltip -> "Thunk"}, "Toilet flush" -> 
       {Tooltip -> "Toilet flush"}, "Rattle (instrument)" -> 
       {Tooltip -> "Rattle (instrument)"}, "Plop" -> {Tooltip -> "Plop"}, 
      "Train horn" -> {Tooltip -> "Train horn"}, "Owl" -> {Tooltip -> "Owl"}, 
      "Cough" -> {Tooltip -> "Cough"}, "Reversing beeps" -> 
       {Tooltip -> "Reversing beeps"}, "Cash register" -> 
       {Tooltip -> "Cash register"}, "Cornet" -> {Tooltip -> "Cornet"}, 
      "Chime" -> {Tooltip -> "Chime"}, "Orchestra" -> 
       {Tooltip -> "Orchestra"}, "Slap, smack" -> {Tooltip -> "Slap, smack"}, 
      "Clarinet" -> {Tooltip -> "Clarinet"}, "Frying (food)" -> 
       {Tooltip -> "Frying (food)"}, "Waves, surf" -> 
       {Tooltip -> "Waves, surf"}, "Field recording" -> 
       {Tooltip -> "Field recording"}, "Sidetone" -> {Tooltip -> "Sidetone"}, 
      "Distortion" -> {Tooltip -> "Distortion"}, "Dial tone" -> 
       {Tooltip -> "Dial tone"}, "Hands" -> {Tooltip -> "Hands"}, 
      "Shuffling cards" -> {Tooltip -> "Shuffling cards"}, 
      "Hiss" -> {Tooltip -> "Hiss"}}}], "LearnedClasses" -> 
  {"Speech", "Male speech, man speaking", "Female speech, woman speaking", 
   "Child speech, kid speaking", "Conversation", "Narration, monologue", 
   "Babbling", "Speech synthesizer", "Shout", "Bellow", "Whoop", "Yell", 
   "Battle cry", "Children shouting", "Screaming", "Whispering", "Laughter", 
   "Baby laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle", 
   "Crying, sobbing", "Baby cry, infant cry", "Whimper", "Wail, moan", 
   "Sigh", "Singing", "Choir", "Yodeling", "Chant", "Mantra", "Male singing", 
   "Female singing", "Child singing", "Synthetic singing", "Rapping", 
   "Humming", "Groan", "Grunt", "Whistling", "Breathing", "Wheeze", 
   "Snoring", "Gasp", "Pant", "Snort", "Cough", "Throat clearing", "Sneeze", 
   "Sniff", "Run", "Shuffle", "Walk, footsteps", "Chewing, mastication", 
   "Biting", "Gargling", "Stomach rumble", "Burping, eructation", "Hiccup", 
   "Fart", "Hands", "Finger snapping", "Clapping", "Heart sounds, heartbeat", 
   "Heart murmur", "Cheering", "Applause", "Chatter", "Crowd", 
   "Hubbub, speech noise, speech babble", "Children playing", "Animal", 
   "Domestic animals, pets", "Dog", "Bark", "Yip", "Howl", "Bow-wow", 
   "Growling", "Whimper (dog)", "Cat", "Purr", "Meow", "Hiss", "Caterwaul", 
   "Livestock, farm animals, working animals", "Horse", "Clip-clop", 
   "Neigh, whinny", "Cattle, bovinae", "Moo", "Cowbell", "Pig", "Oink", 
   "Goat", "Bleat", "Sheep", "Fowl", "Chicken, rooster", "Cluck", 
   "Crowing, cock-a-doodle-doo", "Turkey", "Gobble", "Duck", "Quack", 
   "Goose", "Honk", "Wild animals", "Roaring cats (lions, tigers)", "Roar", 
   "Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet", 
   "Squawk", "Pigeon, dove", "Coo", "Crow", "Caw", "Owl", "Hoot", 
   "Bird flight, flapping wings", "Canidae, dogs, wolves", 
   "Rodents, rats, mice", "Mouse", "Patter", "Insect", "Cricket", "Mosquito", 
   "Fly, housefly", "Buzz", "Bee, wasp, etc.", "Frog", "Croak", "Snake", 
   "Rattle", "Whale vocalization", "Music", "Musical instrument", 
   "Plucked string instrument", "Guitar", "Electric guitar", "Bass guitar", 
   "Acoustic guitar", "Steel guitar, slide guitar", 
   "Tapping (guitar technique)", "Strum", "Banjo", "Sitar", "Mandolin", 
   "Zither", "Ukulele", "Keyboard (musical)", "Piano", "Electric piano", 
   "Organ", "Electronic organ", "Hammond organ", "Synthesizer", "Sampler", 
   "Harpsichord", "Percussion", "Drum kit", "Drum machine", "Drum", 
   "Snare drum", "Rimshot", "Drum roll", "Bass drum", "Timpani", "Tabla", 
   "Cymbal", "Hi-hat", "Wood block", "Tambourine", "Rattle (instrument)", 
   "Maraca", "Gong", "Tubular bells", "Mallet percussion", 
   "Marimba, xylophone", "Glockenspiel", "Vibraphone", "Steelpan", 
   "Orchestra", "Brass instrument", "French horn", "Trumpet", "Trombone", 
   "Bowed string instrument", "String section", "Violin, fiddle", 
   "Pizzicato", "Cello", "Double bass", 
   "Wind instrument, woodwind instrument", "Flute", "Saxophone", "Clarinet", 
   "Harp", "Bell", "Church bell", "Jingle bell", "Bicycle bell", 
   "Tuning fork", "Chime", "Wind chime", "Change ringing (campanology)", 
   "Harmonica", "Accordion", "Bagpipes", "Didgeridoo", "Shofar", "Theremin", 
   "Singing bowl", "Scratching (performance technique)", "Pop music", 
   "Hip hop music", "Beatboxing", "Rock music", "Heavy metal", "Punk rock", 
   "Grunge", "Progressive rock", "Rock and roll", "Psychedelic rock", 
   "Rhythm and blues", "Soul music", "Reggae", "Country", "Swing music", 
   "Bluegrass", "Funk", "Folk music", "Middle Eastern music", "Jazz", 
   "Disco", "Classical music", "Opera", "Electronic music", "House music", 
   "Techno", "Dubstep", "Drum and bass", "Electronica", 
   "Electronic dance music", "Ambient music", "Trance music", 
   "Music of Latin America", "Salsa music", "Flamenco", "Blues", 
   "Music for children", "New-age music", "Vocal music", "A capella", 
   "Music of Africa", "Afrobeat", "Christian music", "Gospel music", 
   "Music of Asia", "Carnatic music", "Music of Bollywood", "Ska", 
   "Traditional music", "Independent music", "Song", "Background music", 
   "Theme music", "Jingle (music)", "Soundtrack music", "Lullaby", 
   "Video game music", "Christmas music", "Dance music", "Wedding music", 
   "Happy music", "Funny music", "Sad music", "Tender music", 
   "Exciting music", "Angry music", "Scary music", "Wind", "Rustling leaves", 
   "Wind noise (microphone)", "Thunderstorm", "Thunder", "Water", "Rain", 
   "Raindrop", "Rain on surface", "Stream", "Waterfall", "Ocean", 
   "Waves, surf", "Steam", "Gurgling", "Fire", "Crackle", "Vehicle", 
   "Boat, Water vehicle", "Sailboat, sailing ship", "Rowboat, canoe, kayak", 
   "Motorboat, speedboat", "Ship", "Motor vehicle (road)", "Car", 
   "Vehicle horn, car horn, honking", "Toot", "Car alarm", 
   "Power windows, electric windows", "Skidding", "Tire squeal", 
   "Car passing by", "Race car, auto racing", "Truck", "Air brake", 
   "Air horn, truck horn", "Reversing beeps", 
   "Ice cream truck, ice cream van", "Bus", "Emergency vehicle", 
   "Police car (siren)", "Ambulance (siren)", 
   "Fire engine, fire truck (siren)", "Motorcycle", 
   "Traffic noise, roadway noise", "Rail transport", "Train", 
   "Train whistle", "Train horn", "Railroad car, train wagon", 
   "Train wheels squealing", "Subway, metro, underground", "Aircraft", 
   "Aircraft engine", "Jet engine", "Propeller, airscrew", "Helicopter", 
   "Fixed-wing aircraft, airplane", "Bicycle", "Skateboard", "Engine", 
   "Light engine (high frequency)", "Dental drill, dentist's drill", 
   "Lawn mower", "Chainsaw", "Medium engine (mid frequency)", 
   "Heavy engine (low frequency)", "Engine knocking", "Engine starting", 
   "Idling", "Accelerating, revving, vroom", "Door", "Doorbell", "Ding-dong", 
   "Sliding door", "Slam", "Knock", "Tap", "Squeak", 
   "Cupboard open or close", "Drawer open or close", 
   "Dishes, pots, and pans", "Cutlery, silverware", "Chopping (food)", 
   "Frying (food)", "Microwave oven", "Blender", "Water tap, faucet", 
   "Sink (filling or washing)", "Bathtub (filling or washing)", "Hair dryer", 
   "Toilet flush", "Toothbrush", "Electric toothbrush", "Vacuum cleaner", 
   "Zipper (clothing)", "Keys jangling", "Coin (dropping)", "Scissors", 
   "Electric shaver, electric razor", "Shuffling cards", "Typing", 
   "Typewriter", "Computer keyboard", "Writing", "Alarm", "Telephone", 
   "Telephone bell ringing", "Ringtone", "Telephone dialing, DTMF", 
   "Dial tone", "Busy signal", "Alarm clock", "Siren", "Civil defense siren", 
   "Buzzer", "Smoke detector, smoke alarm", "Fire alarm", "Foghorn", 
   "Whistle", "Steam whistle", "Mechanisms", "Ratchet, pawl", "Clock", 
   "Tick", "Tick-tock", "Gears", "Pulleys", "Sewing machine", 
   "Mechanical fan", "Air conditioning", "Cash register", "Printer", 
   "Camera", "Single-lens reflex camera", "Tools", "Hammer", "Jackhammer", 
   "Sawing", "Filing (rasp)", "Sanding", "Power tool", "Drill", "Explosion", 
   "Gunshot, gunfire", "Machine gun", "Fusillade", "Artillery fire", 
   "Cap gun", "Fireworks", "Firecracker", "Burst, pop", "Eruption", "Boom", 
   "Wood", "Chop", "Splinter", "Crack", "Glass", "Chink, clink", "Shatter", 
   "Liquid", "Splash, splatter", "Slosh", "Squish", "Drip", "Pour", 
   "Trickle, dribble", "Gush", "Fill (with liquid)", "Spray", 
   "Pump (liquid)", "Stir", "Boiling", "Sonar", "Arrow", 
   "Whoosh, swoosh, swish", "Thump, thud", "Thunk", "Electronic tuner", 
   "Effects unit", "Chorus effect", "Basketball bounce", "Bang", 
   "Slap, smack", "Whack, thwack", "Smash, crash", "Breaking", "Bouncing", 
   "Whip", "Flap", "Scratch", "Scrape", "Rub", "Roll", "Crushing", 
   "Crumpling, crinkling", "Tearing", "Beep, bleep", "Ping", "Ding", "Clang", 
   "Squeal", "Creak", "Rustle", "Whir", "Clatter", "Sizzle", "Clicking", 
   "Clickety-clack", "Rumble", "Plop", "Jingle, tinkle", "Hum", "Zing", 
   "Boing", "Crunch", "Silence", "Sine wave", "Harmonic", "Chirp tone", 
   "Sound effect", "Pulse", "Inside, small room", 
   "Inside, large room or hall", "Inside, public space", 
   "Outside, urban or manmade", "Outside, rural or natural", "Reverberation", 
   "Echo", "Noise", "Environmental noise", "Static", "Mains hum", 
   "Distortion", "Sidetone", "Cacophony", "White noise", "Pink noise", 
   "Throbbing", "Vibration", "Television", "Radio", "Field recording"}|>
