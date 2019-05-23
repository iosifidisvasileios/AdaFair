import matplotlib
matplotlib.use('Agg')

import numpy
import matplotlib.pyplot as plt


def plot_my_results(output_dir, dataset):
    colors = ['b','g','r','c','m','y','k', 'dimgray']
    colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#2ecc71", "#34495e"]
    # bank dataset
    size =7
    if dataset == "bank":
        bar_width = 0.17
        font_size = 11
        names = ['Zafar et al.', 'Krasanakis et al.', 'AdaBoost', 'SMOTEBoost','AdaFair']

        accuracy_list = [0.9014654090146542,0.8986906 , 0.8997750224977501,0.8957604239576042, .87381761823817 ]
        std_accuracy_list = [0.0020856491649241297,0.000369014633856, 0.002307587474996224,0.001978311708861708, .012432088]
        balanced_accuracy_list =[0.6569178324659456, 0.64968265, 0.6884338555612595, 0.7416729802389698, 0.786019 ]
        std_balanced_accuracy_list =[0.004124698886348359, 0.003130033274584, 0.006112057444927729, 0.004371430, .020544993]
        fairness_list =[0.026247620510718384, 0.029363, 0.11459964385212498, 0.1176812405, .0305449 ]
        std_fairness_list =[0.013302190089989351, 0.01322239448814, 0.039373123904823705, 0.0451724263, 0.017990 ]

        tpr_protected_list =[0.3404819694085027, 0.3399602, 0.3769288688278519, 0.50537048, 0.6737057 ]
        std_tpr_protected_list= [0.013168271097689893, 0.027213980849924, 0.019418388890897493, 0.0186173945755, 0.04565]
        tpr_non_protected_list =[0.33533764082441775, 0.3228484, 0.46463747980275044, 0.5921248758, .668128 ]
        std_tpr_non_protected_list= [0.0170381072064758, 0.048545932917187, 0.025794234314879837, 0.0264831284448, 0.043146 ]
        tnr_protected_list= [0.9772712011898972, 0.969662, 0.96924162038161381, 0.951712481, 0.9073496467 ]
        std_tnr_protected_list =[0.0010575422085822227, 0.005373073189526, 0.002439124361957216, 0.0031069363, 0.021830 ]
        tnr_non_protected_list =[0.9715604016454734, 0.9798354, 0.9455251709389116, 0.9207856,0.885394232914 ]
        std_tnr_non_protected_list= [0.002640862219941835,0.005692142153882, 0.00556099856681456, 0.0065386, 0.0238926]

    elif dataset == "compass":
        names = ['Zafar et al.', 'Krasanakis et al.', 'AdaBoost', 'SMOTEBoost','AdaFair']
        bar_width = 0.17
        font_size = 11

        accuracy_list =[0.6442970822281168, 0.6709576, 0.6712769988632057,  0.6547934, .6471011746]
        std_accuracy_list =[0.005771835723483403,0.002795755050071, 0.010489284225590305,  0.0096057431637, 0.010058541]
        balanced_accuracy_list = [0.6357560937166155, 0.6594696875,0.6675175086392817, 0.659335320,.654489 ]
        std_balanced_accuracy_list = [0.006220496647839287, 0.002896127500646,0.010979180765098788, 0.0087638, 0.0105]
        fairness_list =[0.09359745167922937, 0.0424834, 0.3612745751493023, 0.385127848, 0.050852]
        std_fairness_list =[0.028390909071829718, 0.019471642566238, 0.11452832362515976, 0.0983701309, .030084]

        tpr_protected_list =[0.4549621658317228, 0.5501926, 0.41987127961375315, 0.55173150, 0.65728]
        std_tpr_protected_list=[0.0393259444451863, 0.112697835143546, 0.06886461753830742, 0.05969444,0.05791 ]
        tpr_non_protected_list =[0.49193300734626444, 0.5724758, 0.6374989253709892, 0.756840700, 0.675284]
        std_tpr_non_protected_list=[0.030444830417997913, 0.13459096214682, 0.01740228200829841, 0.01695264, 0.043681 ]
        tnr_protected_list=[0.7560185338901767, 0.7689726,  0.8408247749911053, 0.73022974, 0.60997]
        std_tnr_protected_list =[0.032014630828735395, 0.09271311259059, 0.043274616827514024, 0.05730233, 0.02619]
        tnr_non_protected_list =[0.7935183026247083, 0.76223775, 0.6971778455990392, 0.5502110898, .62879051]
        std_tnr_non_protected_list=[0.03743195061831913, 0.10889897263139, 0.02245981608568674, 0.029772, 0.035784]
    #
    elif dataset == "adult":
        names = ['Zafar et al.', 'Krasanakis et al.', 'AdaBoost', 'SMOTEBoost','AdaFair']
        bar_width = 0.17
        font_size = 11

        accuracy_list =[0.8353046936807845, 0.8272768, 0.8475572663022305, .8110412608464672,0.8334735257658934]
        std_accuracy_list =[0.0013164209815277018, 0.00033013739564, 0.0016006291589892912, .0036642513635920214,0.00890679143447399 ]
        balanced_accuracy_list = [0.7096840514775167, 0.70314485, 0.7336316139467047, 0.8047966044212293, 0.7828964294538823]
        std_balanced_accuracy_list =[0.0041676454503743595, 0.002760948261015,  0.0045062591280023954, 0.01699141279896893,  0.009487467261529667]
        fairness_list = [0.13379807768266622, 0.051112,0.2228647743568891, .4781008486511623, .083534562456318]
        std_fairness_list =[0.009987622284623092, 0.009158556474685, 0.017486347288878254,.024084520244839144, 0.0203518118]

        tpr_protected_list =[0.5591357574100587, 0.4361816, 0.3625298364804249, 0.6030174729012379, .720928125]
        std_tpr_protected_list= [0.013522423527874023, 0.035273984476881, 0.018062454772540468, 0.0194391228907989, 0.0573654]
        tpr_non_protected_list =[0.4427906596319713, 0.4565824,0.5330325825505013, 0.81307262231062,0.675974339 ]
        std_tpr_non_protected_list= [0.010025543102808662, 0.049284699149844, 0.011479745452822406, 0.006671775632, 0.04230408]
        tnr_protected_list=[0.970002907720055, 0.9436514, 0.9919610963346415, 0.9555069542195456,  0.906922]
        std_tnr_protected_list =[0.0010566979231614421, 0.02552339920701, 0.001037263342282705, 0.0035842573853, 0.022242]
        tnr_non_protected_list = [0.9525499278154762, 0.976164,0.9395990680478288, 0.708590765221515, 0.8683419375]
        std_tnr_non_protected_list=[0.003620189603987732, 0.007641821041732,0.0038272631907209716, 0.004438476042, 0.02909538]


    elif dataset == "kdd":
        names = [ 'Krasanakis et al.', 'AdaBoost', 'SMOTEBoost','AdaFair']
        bar_width = 0.17
        font_size = 11
        colors = [ "#3498db", "#95a5a6", "#e74c3c", "#2ecc71", "#34495e"]

        accuracy_list = [0.9462, 0.9510608581757918, .9432683119, 0.8653020856304672]
        std_accuracy_list =[0.008, 0.000403011679569436,.008988, 0.009482596863320694]
        balanced_accuracy_list= [0.594622,  0.6671950236883214, 0.7716979650, 0.8450236983993061]
        std_balanced_accuracy_list =  [0.004, 0.0031339775011954506, 0.005451, 0.0016358671824901515]
        fairness_list=  [0.0181234, 0.282380954017813, 0.367454222, 0.0221882142186136]
        std_fairness_list = [0.007231, 0.01181685740347354, 0.01790173, 0.009595585173706027]
        tpr_protected_list= [0.1862, 0.13170866168306072, 0.32000964, 0.8242686431935535]
        std_tpr_protected_list = [0.02111, 0.007100406699076609, 0.0016652, 0.02345361698685365]
        tpr_non_protected_list = [0.1998, 0.4000030727538516, 0.6449769, 0.8212118652972811]
        std_tpr_non_protected_list = [0.02, 0.007807929267985162, 0.0166996, 0.010756865637867031]
        tnr_protected_list = [0.9968, 0.997993973529316, 0.9875248, 0.87299393692887]
        std_tnr_protected_list = [0.0001, 0.0001707044931205857, 0.0004727, 0.01331858107908297]
        tnr_non_protected_list = [0.9957, 0.9839074305822939, 0.94503790, 0.86253327839291]
        std_tnr_non_protected_list = [ 0.003, 0.0012323005001142508, 0.00334006, 0.0085747375]

    elif dataset == "single_adult":
        names = ['AdaFair NoCumul', 'AdaFair']
        size = 4
        font_size = 10
        bar_width = 0.3

        colors = [ "#34495e", "#2ecc71", "#34495e", "#2ecc71"]
        # accuracy_list = [0.7777090014159209, 0.8334735257658934]
        # std_accuracy_list = [0.030064684417580895, 0.008906791434473992]
        # balanced_accuracy_list = [0.7987636933979619, 0.7828964294538823]
        # std_balanced_accuracy_list = [0.0074068468263909235, 0.009487467261529667]
        fairness_list = [0.6166000122397814, 0.08353456245631832]
        std_fairness_list = [0.1579182551466377, 0.01710448303596057]
        tpr_protected_list = [0.5899305758936734, 0.7209281259798667]
        std_tpr_protected_list = [0.042497513537486646, 0.05736540074154048]
        tpr_non_protected_list = [0.8841626189703365, 0.6759743395382416]
        std_tpr_non_protected_list = [0.07939441695204563, 0.042304083509469]
        tnr_protected_list = [0.9560829603525566, 0.9069227135198721]
        std_tnr_protected_list = [0.01124469775573034, 0.023874255078996088]
        tnr_non_protected_list = [0.6337149911894382, 0.8683419375051787]
        std_tnr_non_protected_list = [0.09829222263484681, 0.027108668687253448]

    elif dataset == "single_compass":
        names = ['AdaFair NoCumul', 'AdaFair']
        size = 4
        font_size = 10
        bar_width = 0.3
        colors = [ "#34495e", "#2ecc71", "#34495e", "#2ecc71"]
        # accuracy_list = [0.6686813186813187, 0.6471011746873817]
        # std_accuracy_list = [0.007496020990294776, 0.01355678435872223]
        # balanced_accuracy_list = [0.6658623366276265, 0.6546584411236644]
        # std_balanced_accuracy_list = [0.007604180285551037, 0.011478169183096694]
        fairness_list = [0.2272840304128841, 0.05633060290232009]
        std_fairness_list = [0.07506351872314813, 0.0390507351376023]
        tpr_protected_list = [0.5069406199462202, 0.6656682802275413]
        std_tpr_protected_list = [0.0381354508039561, 0.05032129369256071]
        tpr_non_protected_list = [0.6398830105321432, 0.6831817894682468]
        std_tpr_non_protected_list = [0.03720199511028842, 0.0606794692145965]
        tnr_protected_list = [0.7839015673753595, 0.622210869081045]
        std_tnr_protected_list = [0.02934122119890074, 0.07149642956701875]
        tnr_non_protected_list = [0.6895599275483985, 0.6308168614291902]
        std_tnr_non_protected_list = [0.03305301207796616, 0.06827488636484708]


    elif dataset == "single_bank":
        names = ['AdaFair NoCumul', 'AdaFair']
        font_size = 10
        bar_width = 0.3
        size = 4
        colors = [ "#34495e", "#2ecc71", "#34495e", "#2ecc71"]
        # accuracy_list = [0.8085591440855915, 0.8781796820317969]
        # std_accuracy_list = [0.02354999414032148, 0.014524208278431245]
        # balanced_accuracy_list = [0.8169505526843439, 0.7860198481401538]
        # std_balanced_accuracy_list = [0.005353024009187448, 0.02525125603579959]
        fairness_list = [0.2059109064276321, 0.03594872050013955]
        std_fairness_list = [0.09393951064856917, 0.011628147572499576]
        tpr_protected_list = [0.7960659163308293, 0.6737057522713018]
        std_tpr_protected_list = [0.03396094265151269, 0.06565841038152437]
        tpr_non_protected_list = [0.873800019926338, 0.6681283470427763]
        std_tpr_non_protected_list = [0.049172836565705626, 0.06314609138218173]
        tnr_protected_list = [0.8453768252367302, 0.9073496467673461]
        std_tnr_protected_list = [0.019853402510626564, 0.021830312852951377]
        tnr_non_protected_list = [0.7172000224046069, 0.8853942329145745]
        std_tnr_non_protected_list = [0.069610195263599, 0.02788842925696522]



    elif dataset == "single_kdd":
        font_size = 10
        size = 4
        bar_width = 0.3
        # accuracy_list = [0.8673877160976458, 0.8696731554432884]
        # std_accuracy_list = [0.032032904980520305, 0.0025379469804802084]
        # balanced_accuracy_list = [0.8463746882837765, 0.8437504114120523]
        # std_balanced_accuracy_list = [0.008584368171809742, 0.0047652299777409786]
        fairness_list = [0.4680624163043136, 0.01867421304132011]
        std_fairness_list = [0.094388245413456577, 0.01915261059896256]
        tpr_protected_list = [0.589716843162476, 0.8164428060442368]
        std_tpr_protected_list = [0.10683488097545502, 0.0211570917585225224]
        tpr_non_protected_list = [0.8856265194699513, 0.8135456577952849]
        std_tpr_non_protected_list = [0.0417006443841586, 0.009133166374557297]
        tnr_protected_list = [0.9493233312180405, 0.878120674158547]
        std_tnr_protected_list = [0.01899448967905154, 0.0113788116145074802]
        tnr_non_protected_list = [0.7771705912212024, 0.8677367192503067]
        std_tnr_non_protected_list = [0.059740480856891376, 0.0134335976943407065]
        names = ['AdaFair NoCumul', 'AdaFair']
        colors = [ "#34495e", "#2ecc71", "#34495e", "#2ecc71"]



    plt.figure(figsize=(size, size))
    plt.rcParams.update({'font.size': 11})
    plt.ylim([0,1])
    plt.yticks(numpy.arange(0, 1.00001, step=0.05))
    # plt.legend('center left')
    plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')

    plt.grid(True, axis='y')
    # index = numpy.arange(7)
    # bar_width = 0.17



    # index = numpy.arange(0, 7, step=1)
    # for i in range(0, len(names)):
    #     plt.bar(index + bar_width * i, [accuracy_list[i], balanced_accuracy_list[i], fairness_list[i], tpr_protected_list[i],
    #                                     tpr_non_protected_list[i], tnr_protected_list[i], tnr_non_protected_list[i]]
    #             , bar_width,
    #             yerr=[std_accuracy_list[i], std_balanced_accuracy_list[i], std_fairness_list[i],
    #                   std_tpr_protected_list[i], std_tpr_non_protected_list[i], std_tnr_protected_list[i],
    #                   std_tnr_non_protected_list[i]],
    #             label=names[i], color=colors[i],edgecolor='black')

    index = numpy.arange(0, 5, step=1)
    plt.xticks(index + .57*bar_width , ('Eq.Odds', 'TPR Prot.', 'TPR Non-Prot.', 'TNR Prot.', 'TNR Non-Prot.'))

    #
    for i in range(0, len(names)):
        plt.bar(index + bar_width * i, [fairness_list[i], tpr_protected_list[i],
                                        tpr_non_protected_list[i], tnr_protected_list[i], tnr_non_protected_list[i]]
                , bar_width,
                yerr=[std_fairness_list[i],
                      std_tpr_protected_list[i], std_tpr_non_protected_list[i], std_tnr_protected_list[i],
                      std_tnr_non_protected_list[i]],
                label=names[i], color=colors[i],edgecolor='black')

    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.105), ncol=2, shadow=False,fancybox=True, framealpha=1.0)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, shadow=False,fancybox=True, framealpha=1.0)
    # plt.ylabel('(%)')
    # plt.title("Performance for " + dataset)
    plt.savefig(output_dir,bbox_inches='tight', dpi=200)


#

if __name__ == '__main__':
    # main("compass-gender")
    # plot_my_results("Images/adult_performance.png", "adult")
    # plot_my_results("Images/bank_performance.png", "bank")
    # plot_my_results("Images/compass_performance.png", "compass")
    # plot_my_results("Images/kdd_performance.png", "kdd")
    plot_my_results("Images/adult_single.png", "single_adult")
    plot_my_results("Images/bank_single.png", "single_bank")
    plot_my_results("Images/compass_single.png", "single_compass")
    plot_my_results("Images/kdd_single.png", "single_kdd")

#