import sklearn
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from joblib import load

st.title("WATER PUMP STATUS PREDICTION USING ML MODEL TRAINED ON TAARIFA DATASET")

st.write("Please select the data related to water pumps from the side bar using which the model will predict the status of the water pump whether it is functional, non-functional or functional but needs repair")

df = {}

st.sidebar.title("Required information related to water pump")

df['latitude'] = st.sidebar.slider("Latitude of water pump location",-11.64944018,-2e-08)

df['longitude'] = st.sidebar.slider("Longitude of water pump location",0.0,40.34519307)

df["quantity"] = st.sidebar.selectbox("Quantity of water",['unknown','dry','insufficient','seasonal','enough'])

df["permit"] = st.sidebar.selectbox("Permit for well available or not",[True,False])

df["public_meeting"] = st.sidebar.selectbox("Public meeting for well conducted or not",[True,False])

df["funder"] = st.sidebar.selectbox("Name of funder",['Roman', 'Grumeti', 'other', 'Unicef', 'Mkinga Distric Coun',
       'Dwsp', 'Rwssp', 'Wateraid', 'Private', 'Danida', 'World Vision',
       'Lawatefuka Water Supply', 'Biore', 'Rudep', 'Hesawa', 'Twe',
       'Isf', 'African Development Bank', 'Government Of Tanzania',
       'Water', 'Private Individual', 'Undp', 'unknown', 'Not Known',
       'Kirde', 'Cefa', 'Ces(gmbh)', 'European Union', 'Lga',
       'District Council', 'Muwsa', 'Dwe/norad', 'Kkkt_makwale',
       'Ces (gmbh)', 'Kkkt', 'Roman Catholic', 'Norad', 'Adra', 'Sema',
       'Dwe', 'Rc Church', 'Swisland/ Mount Meru Flowers', 'Ifad',
       'Swedish', 'Idc', 'He', 'Jica', 'Aict', 'Tcrs', 'Kiuma',
       'Germany Republi', 'Netherlands', 'Nethalan', 'Tasaf',
       'Concern World Wide', 'Wfp', 'Lips', 'Sida', 'World Bank', 'Tanza',
       '0', 'Shipo', 'Fini Water', 'Oxfarm', 'Village Council', 'Wvt',
       'Dhv', 'Ir', 'Oikos E.Afrika', 'Anglican Church', 'Peters',
       'Donor', 'Amref', 'Ministry Of Water', 'Adb', 'Jbg', 'Dadis',
       'Germany', 'Kibaha Town Council', 'Dfid',
       'Rural Water Supply And Sanitat', 'Wananchi', 'Fw', 'No', 'Co',
       'Ridep', 'Tassaf', 'Hans', 'Finw', 'Fin Water', 'Oxfam',
       'Plan International', 'Go', 'Cdtf', 'Shawasa', 'Un', 'Commu',
       'Community', 'Save The Rain Usa', 'Tlc', 'Rc Churc', 'Plan Int',
       'W.B', 'Lvia', 'Songea District Council', 'Hifab', 'Rc Ch', 'Snv',
       'National Rural', 'Is', 'Giz', 'Cspd', 'Wsdp', 'Finn Water',
       'Villagers', 'Ereto', 'Abasia', 'Unhcr', 'Kuwait',
       'Magadini-makiwaru Water', 'Kaemp', 'Rcchurch/cefa', 'Tardo',
       'Sabemo', 'Missi', 'Dmdd', 'Dhv\\norp', 'Mission', 'Ru',
       'Halmashauri Ya Wilaya Sikonge', 'Japan', 'Ki', 'Marafip', 'Ta',
       'Il', 'Bank', 'Ded', 'Sabodo', 'Soda', 'Lwi', 'Ics', 'African',
       'Cipro/government', 'Tabora Municipal Council', 'Hewasa', 'Jaica',
       'Solidarm', 'Kanisa La Menonite', 'Rc', 'Killflora', 'Wua', 'Dw',
       'Md', 'Quwkwin', 'Dh', 'Mbiuwasa', 'Care International', 'Dasip',
       'Hsw', 'Mwaya Mn', 'Tz Japan', 'Concern', 'Caritas', 'Conce',
       'Devon Aid Korogwe', 'Kiliwater', 'Lamp', 'Bsf', 'Bgm', 'Fathe',
       'Unice', 'Songea Municipal Counci', 'Water User As',
       'Islamic Found', 'Vwc', 'Acra', 'Gtz', 'Kuwasa',
       'China Government', 'Churc', 'Mkinga  Distric Cou', 'Cafod',
       'Hw/rc', 'Tacare', 'Urt', 'Water Aid /sema', 'Kilwater', 'Ndrdp',
       'Nethe', 'Adp', 'Holland', 'Cocen', 'Ncaa', 'Finwater', 'Dwssp',
       'The Desk And Chair Foundat', 'Kkkt Church', 'Jika', 'Tuwasa',
       'Irish Ai', 'Mdrdp', 'Kilindi District Co', 'Kidp', 'St',
       'Wd And Id', 'Serikali', 'Po', 'Ga', 'Cocern',
       'Finida German Tanzania Govt', 'Idara Ya Maji', 'Swiss If',
       'Miziriol', 'H', 'Oikos E.Africa/european Union', 'Ilct',
       'Peter Tesha', 'Ms', 'Red Cross', 'Losaa-kia Water Supply',
       'Kanisa Katoliki Lolovoni', 'Tdft', 'Rished', 'Village Government',
       'Cmsr', 'W', 'Partage', 'Missionaries', 'Roman Cathoric-same',
       'Cefa-njombe', 'Aar', 'Mileniam Project', 'Undp/ilo', 'Dads',
       'Luthe', 'Twesa', 'Plan Internatio', 'Solidame',
       'Watu Wa Ujerumani', 'Water Aid/sema', 'Gen', 'Redep',
       'Singida Yetu', 'Tanapa', 'Tanzakesho', 'World Vision/adra',
       'Kalta', 'Kalitasi', 'Wwf', 'Ilo', 'Kidep', 'Ka',
       'Total Land Care', 'Padep', 'I Wash', 'Si', 'Halmashauri',
       'Vifafi', 'Village Community', 'Songas', 'Cmcr', 'Cdcg',
       'Happy Watoto Foundation', 'Cg', 'Tredep', 'Village'])

df["installer"] = st.sidebar.selectbox("Name of installer",['Roman', 'GRUMETI', 'World vision', 'UNICEF', 'Artisan', 'DWE',
       'DWSP', 'Water Aid', 'Private', 'DANIDA', 'Lawatefuka water sup',
       'WEDECO', 'Danid', 'TWE', 'ISF', 'other', 'District council',
       'Water', 'WU', 'unknown', 'Central government', 'CEFA', 'Commu',
       'Accra', 'World Vision', 'LGA', 'MUWSA', 'KKKT _ Konde and DWE',
       'Government', 'KKKT', 'RWE', 'Adra /Community', 'SEMA', 'SHIPO',
       'HESAWA', 'ACRA', 'Community', 'IFAD',
       'Sengerema Water Department', 'HE', 'Kokeni', 'DA', 'Adra', 'AICT',
       'KIUMA', 'CES', 'Adra/Community', 'Hesawa', 'Water board',
       'LOCAL CONTRACT', 'TASAF', 'World', '0', 'Shipo', 'Fini water',
       'OXFARM', 'Villagers', 'Idara ya maji', 'WVT', 'Ir', 'DANID',
       'Angli', 'Amref', 'JBG', 'Dmdd', 'TCRS', 'RC Church', 'WATER AID',
       'JICA', 'Gwasco L', 'AMREF', 'wananchi', 'FW',
       'Central Government', 'MWE &', 'Gove', 'RC CHURCH', 'TDFT',
       'RWE/DWE', 'Central govt', 'World Bank', 'TWESA', 'Norad', 'Hans',
       'FinW', 'FIN WATER', 'OXFAM', 'Plan Internationa',
       'District Council', 'Fini Water', 'Oikos E .Africa', 'SHAWASA',
       'UN', 'NORAD', 'TLC', 'LVIA', 'Rhobi', 'Is', 'KILI WATER',
       'FINN WATER', 'FINI WATER', 'DHV', 'DDCA', 'RWSSP', 'Ce',
       'KYASHA ENTERPR', 'ERETO', 'Villa', 'Priva', 'KUWAIT',
       'Magadini-Makiwaru wa', 'Af', 'RCchurch/CEFA', 'Tardo',
       'GOVERNMENT', 'Individuals', 'Chamavita', 'GEN', 'Missi',
       'DAWASCO', 'Gover', 'Mission', 'Halmashauri ya wilaya sikonge',
       'Ki', 'HAPA SINGIDA', 'Consulting Engineer', 'Co',
       'Handeni Trunk Main(', 'Local technician', 'Centr', 'CONS', 'DW',
       'District water department', 'Sabodo', 'MLADE', 'LWI', 'ICS',
       'DED', 'Kuwait', 'JUIN CO', 'GOVER', 'CIPRO/Government', 'MWE',
       'MTUWASA', 'Unisef', 'Wizara ya maji', 'JAICA', 'MTN', 'Local',
       'RC', 'Distri', 'ADRA', 'DH', 'RC Ch', 'MBIUWASA',
       'Care international', 'CJEJOW CONSTRUCTION', 'Wachina', 'HSW',
       'Communit', 'Go', 'WATERAID', 'Kiliwater', 'TA', 'wanan',
       'Region water Department', 'Ndanda missions',
       'District Water Department', 'Fathe', 'Wa', 'VILLAGE COUNCIL',
       'RDC', 'Local  technician', 'TASSAF', 'VWC', 'TAN PLANT LTD',
       'GTZ', 'KUWASA', 'Hydrotec', 'Pr', 'Ch', 'Jaica', 'BESADA',
       'CBHCC', 'HW/RC', 'CCEC', 'WORLD BANK', 'Water Aid /sema',
       'Kilwater', 'Da', 'District water depar', 'HOLLAND', 'Active MKM',
       'GEOTAN', 'NCAA', 'FinWater', 'The desk and chair foundat',
       'KKKT CHURCH', 'SHY BUILDERS', 'Finwater', 'JIKA', 'DMDD', 'CDTF',
       'KAEMP', 'TUWASA', 'MARAFIP', 'MDRDP', 'Village Council', 'KIDP',
       'Wananchi', 'St', 'WD and ID', 'Po', 'Ga', 'Swiss If', 'Miziriol',
       'H', 'ILCT', 'RED CROSS', 'Losaa-Kia water supp', 'Jica', 'PET',
       'VTECOS', 'Msabi', 'CMSR', 'Do', 'BSF', 'Mileniam project',
       'not known', 'SOLIDAME', 'Tanzania Government', 'ABASIA', 'MUWASA',
       'TANAPA', 'Kalta', 'Kalitasi', 'WWF', 'RIDEP', 'NORA', 'WB',
       'COCANE', 'BGM', 'KA', 'Total land care', 'TAWASA', 'PADEP',
       'CONCERN', 'Save the rain USA', 'Korogwe water works', 'Local te',
       'SI', 'Cosmo', 'Halmashauri', 'Consuting Engineer',
       'JANDU PLUMBER CO', 'Jicks', 'VIFAFI', 'NYAKILANGANI CO', 'Wo',
       'Happy watoto foundation', 'TREDEP', 'KKT', 'GOVERN', 'W/',
       'Serikali'])

df["basin"] = st.sidebar.selectbox("Name of basin",['Lake Nyasa', 'Lake Victoria', 'Pangani',
       'Ruvuma / Southern Coast', 'Internal', 'Lake Tanganyika',
       'Wami / Ruvu', 'Rufiji', 'Lake Rukwa'])

df["subvillage"] = st.sidebar.selectbox("Name of subvillage",['other', 'Majengo', 'Mwabasabi', 'Center', 'Shuleni', 'Karume',
       'Afya', 'Sokoni', 'Marurani Juu', 'Msufini', 'Ulkusare', 'Kihanga',
       'Mtakuja', 'Mapinduzi', 'Moivaro', 'Shule', 'Magwila', 'Juhudi',
       'Maendeleo', "Chang'Ombe", 'Majengo B', 'Msumbiji', 'Mikoroshini',
       'Mbuyuni', 'Mtaa Wa Kitunda Kati', 'Mpakani', 'Madukani',
       'Kijiweni', 'Kati', 'Sekondari', 'Kigamboni', 'Uzunguni',
       'Kanisani', 'I', 'Mjini', 'Stooni', 'Azimio', 'Manzese', 'Isanga',
       'Bagamoyo', 'Kijijini', 'Kawawa', 'Kumsenga', 'Songambele', 'Ccm',
       'Zahanati', 'Kilimanihewa', 'Ujamaa', 'Magomeni', 'Muungano',
       'Mnazi Mmoja', 'Mahenge', 'Juu', 'Barazani', 'Miembeni', 'Ikulu',
       'Muongozo', 'Mwenge', 'unknown', 'Kibaoni', 'Uswahilini',
       'Misufini', 'M', 'Kilimahewa', 'Msikitini', 'Makungu',
       'Mtaa Wa Kivule', 'Mkwajuni', 'Senta', 'Mbugani', 'Mapambano',
       'Ukombozi', 'Bondeni', 'Mchangani', 'S', 'Migombani', 'Mwinyi',
       'Bunukangoma', 'Nyerere', 'Kariakoo', 'Mtaa Wa Kichangani',
       'Mpanda', 'Tankini', 'Vikuge', 'Mjimwema', 'Dodoma', 'Mashariki',
       'Mnazimmoja', 'Gulioni', 'Kabale', 'Mwandu', 'Ofisini', 'Mwamala',
       'Kichangani', 'Nyashimba', 'Kongei', 'Kakola', 'Gezaulole',
       'Kisiwani', 'Relini', 'Misheni', 'Arauyo', 'Kusini', 'Minazini',
       'Mpuguso', 'Bwawani', 'Mtaa Wa Mzinga', 'Mwandege', 'Mtukula',
       'Kisesa', 'Ilala', 'Nyange', 'K', 'Mlimani', 'Maweni', 'K/Center',
       'Msolwa Kati', 'Sokoine', 'Usalama', 'Mwabayanda', 'Njiapanda',
       'Elimu', 'Station', 'Uhuru', 'Jangwani', 'Barabarani', 'Bomani',
       'L', '1', 'Lugombo', 'Mnadani', 'Mabatini', 'Magharibi', 'Tandika',
       'Temeke', 'Stesheni', 'Mission', 'Majimaji', 'Posta', 'Kibaoni B',
       'Lusinde', 'Kiwanjani', 'Bombani', 'Barabara 2', 'Umoja',
       'Magunga', 'Mwangaza', 'Kagera', 'Chemchem', 'Ngelele', 'Katumba',
       'Igalula', 'Darajani', 'Mbuyuni B', 'Mpunguti', 'Nmc',
       'Marurani Kati', 'Namba Moja', 'Kaskazini', 'Zaire', 'Lupaso',
       'Kivukoni', 'Magereza', 'Mtoni', 'Mwaya', 'Kabanga', 'Njia Panda',
       'Masebe', 'U', 'Amani', 'Centre', 'Legeza Mwendo', 'Kiwili',
       'Jitegemee', 'Mtaa Wa Vikongoro', 'Msasani', 'Tanesco', 'Makiungu',
       'Majengo A', 'Barabara 6', 'Makuyuni', 'Ushirika', 'N', 'Mkuyuni',
       'Kidalini', 'Mshikamano', 'Bujingwa', 'Kaloleni', 'Huduma',
       'Mazengo', 'Mwembeni', 'Tangini', 'Stendi', 'Chakwale Mjini',
       'Mvuleni', 'Kizani', 'Maji Coast', 'Mkongo', 'Mtaa Wa Kipunguni B',
       'Mlangali', 'Matale', 'Kasanga', 'Lusungo', 'Mpirani', 'Upendo',
       'Central', 'Mgudeni', 'Mandela'])

df["scheme_name"] = st.sidebar.selectbox("Name of scheme",['Roman', 'unknown', 'other', 'None', "wanging'ombe water supply s",
       'Quick wins Program', 'Komaka mandaka', 'Chalinze wate', 'UNDP',
       'Ngana water supplied scheme', 'Bomala',
       'Kirua kahe pumping water trust', 'Mtwango water supplied sche',
       'K', 'Machumba estate pipe line', 'wangama water supply scheme',
       'Laela group water Supp', 'Makwale water supplied sche',
       'Njoro Water Supply', 'Kirua kahe gravity water supply trust',
       'Mabula mountains spr', 'Mkongoro One',
       'Maambreni gravity water supply', 'M', 'Kaisho/Isingiro w',
       "wanging'ombe supply scheme", 'Tove Mtwango gravity Scheme',
       'Tengeru gravity water supply', 'Loruvani gravity water supply',
       'HESAWA', 'B', 'Chankele/Bubango water project', 'Bagamoyo wate',
       'Kijiji', 'Mradi wa maji wa mpitimbi', 'Shallow well',
       'Losaa-Kia water supply', 'Kiwele', 'N',
       'Tuvaila gravity water supply', 'Kan', 'TASAF', 'Shagai streem',
       'Distri', 'Kanga water supplied scheme',
       'Njalamatata water gravity scheme', 'Mtowisa water suply',
       'Amani spring', 'Meseke', 'Kisimiri gravity water supply', 'S',
       'Hempanga water supply', 'EKTM 3 water supply',
       'Vyama vya watumia maji', 'Sinyanga water supplied sch',
       'Olkokola pipe line', 'Kilotweni water supply', 'Kabali',
       'Kazilankanda Water Supply', 'MWS', 'Dimamba', 'Ma', 'Borehole',
       'Mahuni', 'Ufinga river', 'Mtiro pipeline',
       'Mradi wa maji wa sikonge', 'DANIDA', 'Mtimbira',
       'Kindoroko water supply', 'Kabuye', 'Saitero olosaita pipe line',
       'Otaruni water supply', 'Mowasu', 'Mbokomu east', 'World Bank',
       'Libango water use group scheme', 'Mkongoro Two', 'Government',
       'Nabaiye pipe line', 'Una mkolowoni', 'Kanenge',
       'Tanzania flowers pipe line', 'Kyonza', 'Ikela Wa',
       'Kifaru water Supply', 'Huru mawela water project', 'Kaviwasu',
       'Mradi wa maji wa kilagano', 'Olumulo pipe line',
       'mtwango water supply scheme', 'Marangu baraza',
       'Sinyanga  water supplied sc', 'kaleng', 'Nzi',
       'Uchira water users association', 'Lyamungo umbwe water supply',
       'Mamire water supply', 'Mwaya Mn', 'Kit', 'Kidahwe water project',
       'Majimingi', 'Shagayo forest', 'Nabai pipe line',
       'matembwe water supply schem', 'Mradi wa maji Komuge',
       'Nduruma pipe line', 'Chovora', 'Kibohelo  forest',
       "Uroki-Bomang'ombe water sup", 'Mangamba forest', 'Tangeni',
       'Mlimba W', 'Water from DAWASCO', 'Muwimb',
       'Mradi wa maji wa mahanje', 'Hingilili', 'Saseni', 'RWSSP',
       'Mgaraganza water project', 'Handeni Trunk Main(H', 'Kwa',
       'Mtandao wa Mabomba', 'Nakombo', 'Nyafisi',
       'Chanjare water supply', 'Sabodo Borehole Scheme', 'I', 'Masa',
       'Lupali', 'Vulue water supply', 'Machame water supply',
       'Kongei water supply', 'It', 'Shengui forest',
       'JAICA Borehole Scheme', 'World banks', 'Mradi wa maji Shirati',
       'Mitema', 'Ilolo', 'Cham', 'Magati gravity water', "Nyang'",
       'Matund', 'MAKOGA WATER SUPPLY', 'Kiboelo forest',
       'NCHULOWAIBALE WATER SUPPLY SCHEME', 'Ru', 'Ilente streem',
       'Bagamoyo Wate', 'RUMWAMCH', 'Magang', 'Mang`ula', 'upper Ruvu',
       'Losaa Kia water supply', 'G', 'BRUDER', 'Mgandazi',
       'Boza water supply', 'Mbuo mkunwa water supply',
       'Kagongo water project', 'Anglic', 'Ga',
       'Igongolo gravity water sche', 'LIPS Water Scheme',
       'Mradi wa maji wa peramiho', 'Marangu west', 'Itete wa',
       'Monduli pipe line', 'Maleng', 'Nyachenda', 'none',
       'Mkalamo water supply', 'Central basin', 'Ikonda', 'Mroroma',
       'Robanda pumping scheme', 'Mamsera water supply',
       'Lemanyata pipe line', 'Nselembwe water supply', 'Tove mtwango',
       'TPRI pipe line', 'Mnyawi water supply',
       'Likamba mindeu pipe line', 'Hedaru kati water supply',
       'Maramba gravity spri', 'Maho', 'Kimasaki gravity water supply',
       'imalinyi water supply schem', 'Mkongoro one', 'Igosi',
       'Lake Victoria pipe scheme', 'Mradi wa maji wa matimila',
       'Gamowaso', 'Nyamitoko  water', 'Holili water supply', 'U',
       'Shirimatunda water Supply', 'Lufuo', 'Mradi wa maji wa pito',
       'MONGAHAI RIVER', 'Timbolo sambasha TPRI pipe line', 'Kasangezi',
       'Lyamungo-Umbwe water supply', 'Ki', 'Wasa', 'Kise', 'Uhekule',
       'Nzihi', 'W', 'Mradi wa mkombozi', 'no scheme',
       'Marua msahatie water supply', 'Idodi',
       'Ngamanga water supplied sch', 'Sanje Wa', 'ADP Simbo',
       'shallow well', 'Ihum', 'Nasula gravity water supply', 'Janda',
       'Namahimba Water gravity scheme', 'Mkuzu forest',
       'Mradi wa maji wa Ipole', 'Tove', 'Endawasu', 'Moronga',
       'Murufiti', 'Malemeu gravity water supply', 'Mlomboza forest',
       'Uru shimbwe', 'Nyamtukuza', 'Mi', 'Maboga',
       'Nyangao Water Supply', 'BL Kashashi',
       'Wangingombe gravity Scheme', 'Bangata water project',
       'Vianzi Water Supply', 'Kidia kilemapunda', 'Tutu',
       'ngamanga water supplied sch', 'Du', 'Oldonyowas maji salama',
       'Nyantamba', 'Mvaji ri', 'Churu water supply', 'Kenswa',
       'Mlinga streem', 'Ballaa pipe line', 'Kihoro',
       'Gallapo water supply', 'Ngulu water Supply',
       'Manyoni water supply', 'Ikondo electrical water sch',
       'Mambreni gravity water supply', 'Ushiri water supply',
       'Nameqhwadiba', 'Kalesha water supply',
       'Msitu wa tembo pipe scheme', 'Mushori', 'Gyewasu',
       'Kyamara gravity water supply', 'Ntom', 'Nywelo streem',
       'Msakangoto', 'Mtikanga gravity Scheme', 'Mtam',
       'Mawande gravity Scheme', 'GEN Borehole Scheme',
       'Kasahunga piped scheme', 'D', 'Olkungabo gravity water supply',
       'Riftvalley Project water supply',
       'Maji ya Chai gravity water supply', 'Kigongoi gravity wat',
       'Mkalama Water supply', 'WAUSA', 'Mradi wa maji wa wino',
       'Lema water supplied scheme', 'Mradi wa maji wa mbinga mh',
       'Mlowa', 'Kizingu', 'Ibiki gravity water scheme', 'Ulanda', 'A',
       'Mwazye water supply', 'Mongwa r', 'Ms', "Ng'au",
       'Mukabuye gravity water supply', 'Jongoo',
       'Kabingo/kiobela gravity  water supply', 'Mbakwe water supply',
       'BL Makiwaru', 'Orkesumeti pipe scheme', 'Mgowel',
       'Jumuhiya ya watumia maji', 'Gale water supply',
       'Kagenyi water sup', 'Matala pipeline', 'Ichonde',
       'EKTM 2 water suply', 'Makiyui stream', 'Nyanganga water project',
       'Namanga water project', 'Idunda', 'SHIPO',
       'Mradi wa maji wa maposeni', 'Kilimi and uchama dam',
       'Utengule gravity Scheme', 'Kisuma', 'Mshiri pipeline', 'J',
       'Mseke', 'Lukare water supp', 'Mabira water supp',
       'imalinyi supply scheme', 'Mangil', 'Water AID',
       'Mwao/Mtimue spri', 'Japan', 'Utelewe',
       'Kibondo gravity point source water supply', 'Kafulo water supp',
       'Jumuhiya ya watumia  maji', 'Kabanga', 'MPEISA',
       'Kitumbeine water project', 'Nyarubano', 'Vumamti',
       'Kindoi streem', 'Mlangarini pipe line', 'Mlimani pipe line'])

df["extraction_type"] = st.sidebar.selectbox("Type of extraction used",['gravity', 'submersible', 'swn 80', 'nira/tanira', 'india mark ii',
       'other', 'ksb', 'mono', 'windmill', 'afridev', 'other - rope pump',
       'india mark iii', 'other - swn 81', 'other - play pump', 'cemo',
       'climax', 'walimi'])

df["management"] = st.sidebar.selectbox("Name of management",['vwc', 'wug', 'other', 'private operator', 'water board', 'wua',
       'company', 'water authority', 'parastatal', 'unknown',
       'other - school', 'trust'])

df["payment"] = st.sidebar.selectbox("Type of payment",['pay annually', 'never pay', 'pay per bucket', 'unknown',
       'pay when scheme fails', 'other', 'pay monthly'])

df['water_quality'] = st.sidebar.selectbox("Quality of water",['soft', 'salty', 'milky', 'unknown', 'fluoride', 'coloured',
       'salty abandoned', 'fluoride abandoned']) 

df['source'] = st.sidebar.selectbox("Source of water",['spring', 'rainwater harvesting', 'dam', 'machine dbh', 'other',
       'shallow well', 'river', 'hand dtw', 'lake', 'unknown'])

df['source_class'] = st.sidebar.selectbox("Class of water source",['groundwater', 'surface', 'unknown'])

df['waterpoint_type'] = st.sidebar.selectbox("Type of water point",['communal standpipe', 'communal standpipe multiple', 'hand pump',
       'other', 'improved spring', 'cattle trough', 'dam'])

df['amount_tsh'] = st.sidebar.slider("Total static head of water",0,35000)

df['gps_height'] = st.sidebar.slider("GPS height of water pump location",-90,2770)

df['region_code'] = st.sidebar.slider("Region code of water pump location",1,99)

df['population'] = st.sidebar.slider("Population of water pump location",0,30500)

df['date_recorded'] = st.sidebar.date_input("Date when the pump data was recorded")

df["construction_year"] = st.sidebar.number_input("Year when the pump was constructed",min_value=1900,step=1)

st.header("Location of selected water pump in map")

loc = pd.DataFrame({"latitude":[df["latitude"]],"longitude":[df["longitude"]]})

st.map(loc)

data = {}

#data preprocessing for input to model
df["location"] = np.sqrt(df["latitude"]**2+df["longitude"]**2)

df["age_in_years"] = df["date_recorded"].year - df["construction_year"]

def col_std(feature):
	std = load("std_"+feature+".joblib")
	data[feature] = std.transform(np.array(df[feature]).reshape(-1,1))[0][0]

features = ['amount_tsh','region_code','location','age_in_years','population','gps_height']
for feature in features:
	col_std(feature)

q_feat = ['unknown','dry','insufficient','seasonal','enough']
for i in range(len(q_feat)):
	if df["quantity"] == q_feat[i]:
		data["quantity_enc"] = [i]

if df["public_meeting"]==True:
	data["public_meeting"] = [1]
else:
	data["public_meeting"] = [0]
    
if df["permit"]==True:
	data["permit"] = [1]
else:
	data["permit"] = [0]

data = pd.DataFrame(data)

def ohe(cat_feat):
	ohenc = load("ohenc_"+cat_feat+".joblib")
	if df[cat_feat] not in ohenc.categories_[0]:
		data[cat_feat] = "other"
	else:
		data[cat_feat] = df[cat_feat]
	return ohenc.transform(np.array(data[cat_feat]).reshape(-1,1))
categorical_features = ["funder","installer","basin","subvillage","scheme_name","extraction_type","management","payment",
                           "water_quality","source","source_class","waterpoint_type"]

for cat_feat in categorical_features:
	globals()[f"x_{cat_feat}"] = ohe(cat_feat)

num_feat = np.array([data['permit'][0],data['public_meeting'][0],data['amount_tsh'][0],data['gps_height'][0],data['region_code'][0],data['population'][0],data['quantity_enc'][0],data['location'][0],data['age_in_years'][0]])

from scipy.sparse import hstack

x_enc = hstack((x_funder,x_installer,x_basin,x_subvillage,x_scheme_name,x_extraction_type,x_management,x_payment,x_water_quality,x_source,x_source_class,x_waterpoint_type,num_feat)).tocsr()

st.header("Predicting the status of water pump")
st.write("After selection of pump related data in side bar the predicted status of pump will be displayed below")

def pred(x):
	rf_model = load("RF_model.joblib")
	rf_pred = rf_model.predict(x)
	if rf_pred == 0:
		rf_ps = "functional"
	elif rf_pred == 1:
		rf_ps = "functional needs repair"
	else:
		rf_ps = 'non functional'
	st.write("Pump status predicted by ML model :",rf_ps)

pred(x_enc)
