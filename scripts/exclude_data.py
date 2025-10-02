"""Defines which data to exclude for TICs with problematic sectors."""

from functools import partialmethod

from sqlalchemy import select, delete

from cache_interface import ExcludeDataTable, CacheSession

# TODO: for 260161144 get independent constraint from remaining sectors

_manual_exclude_data = {
    91961: ["QLP"],
    2353789: ["QLP"],
    3815360: ["OOE"],
    3816260: ["OOE"],
    3921749: ["OOE"],
    5092088: ["OOE"],
    5205367: ["OOE"],
    8592033: [51, 78],
    8624617: [78, 79],
    9054370: ["OOE"],
    9381557: ["OOE"],
    9433212: ["OOE"],
    11704852: [15],
    12029932: ["OOE"],
    13092260: ["OOE"],
    13037534: ["OOE"],
    13974582: ["OOE"],
    16479253: ["BLSOOE"],
    16728252: ["OOE"],
    16805617: ["OOE"],
    17451194: [15],
    22146154: ["QLP"],
    23555025: ["OOE"],
    24004619: ["OOE"],
    24133365: [12],
    24433067: [6, 32, "OOE"],
    24935204: ["OOE"],
    26489741: ["OOE"],
    26568753: ["QLP"],
    27767184: ["OOE"],
    27843942: ["OOE"],
    27915909: ["QLP"],
    28351918: ["OOE"],
    29914747: ["QLP"],
    30382918: ["QLP"],
    30993392: ["OOE"],
    31810287: ["OOE"],
    32000625: ["QLP"],
    32449963: ["OOE"],
    33912852: ["OOE"],
    35349987: ["QLP"],
    37435039: ["OOE"],
    41561453: ["QLP"],
    42827646: [87, "OOE"],
    48356677: ["OOE"],
    48507019: ["OOE"],
    48658291: ["OOE"],
    50849800: ["QLP"],
    52280468: ["QLP"],
    52924466: ["QLP"],
    53085065: ["OOE"],
    53847727: ["OOE"],
    55252407: [61, 62],
    55489734: ["OOE"],
    55497281: ["OOE"],
    58466459: ["OOE"],
    60284401: ["OOE"],
    60979860: ["OOE"],
    61724376: ["OOE"],
    62762261: ["OOE"],
    63071165: ["OOE"],
    63074282: [15],
    63315565: ["OOE"],
    63579446: ["OOE"],
    63972440: [56],
    64904640: ["QLP"],
    65628544: ["OOE"],
    66355834: ["OOE"],
    66545120: [16, 77],
    66602813: ["OOE"],
    66786762: ["OOE"],
    67719432: ["QLP"],
    69949079: ["QLP"],
    71531463: [74],
    74528318: ["QLP"],
    74528791: [9, 35, 36, "OOE"],
    75969405: ["QLP"],
    77030554: ["OOE"],
    77557473: ["QLP"],
    78151317: ["OOE"],
    78450017: [6],
    78817217: ["QLP", 87],
    80025900: [9, 34, 35],
    80650858: ["QLP"],
    80929740: ["OOE"],
    82362447: [35],
    83391903: ["OOE"],
    84069091: ["OOE"],
    84628518: [41],
    86232609: ["OOE"],
    87251422: ["OOE"],
    89018905: ["QLP"],
    89522181: [75, 82],
    89787778: ["QLP"],
    99028319: ["QLP"],
    99028324: ["QLP"],
    92743594: ["QLP"],
    92902951: ["QLP"],
    95867802: ["QLP"],
    99813351: ["OOE"],
    100029948: ["QLP"],
    103683084: ["OOE"],
    106055469: ["QLP"],
    111267340: ["QLP"],
    112099606: ["OOE"],
    120258315: ["OOE"],
    122285268: ["QLP"],
    122446076: ["OOE"],
    123222370: ["QLP"],
    124350360: ["QLP"],
    126336928: ["OOE"],
    126944775: [68],
    127365016: ["OOE"],
    128721058: [83],
    130158361: ["OOE"],
    137503140: ["OOE"],
    137975907: ["OOE"],
    138650903: ["OOE"],
    138951779: ["OOE"],
    139188326: ["OOE"],
    139721086: [67],
    139732428: ["QLP"],
    141148944: [75],
    142742998: ["OOE"],
    142865103: ["OOE"],
    144000467: ["OOE"],
    144539611: ["OOE"],
    145762697: [8, "QLP"],
    148612685: [9],
    149030418: ["QLP"],
    149319411: ["OOE", "QLP"],
    149861542: ["OOE"],
    152223725: ["QLP"],
    152825521: ["OOE"],
    153709888: ["OOE"],
    153742549: ["OOE"],
    153987857: ["OOE"],
    157754133: ["OOE"],
    157968186: ["OOE"],
    158582801: ["OOE"],
    158635832: [81],
    158660631: ["OOE"],
    159720778: [15],
    159723646: ["QLP"],
    160085375: ["OOE"],
    160328766: ["OOE"],
    160472514: ["OOE"],
    161668117: ["OOE"],
    162035117: ["OOE"],
    164781045: ["OOE"],
    166090445: ["OOE"],
    167756615: ["OOE"],
    168426455: ["QLP"],
    169819162: [14, 15],
    170344769: [14, 15],
    170635351: ["QLP"],
    172171873: ["OOE"],
    172462064: ["OOE"],
    172517058: ["OOE"],
    172529744: ["OOE"],
    172742334: ["OOE"],
    172933154: ["OOE"],
    173756896: ["QLP"],
    173797509: ["OOE"],
    175969976: [61],
    176937887: ["OOE"],
    178284729: ["OOE"],
    178857934: ["OOE"],
    179410793: ["OOE"],
    180412528: ["OOE"],
    181862539: ["OOE"],
    183469904: [8],
    184739277: ["QLP"],
    191463077: ["OOE"],
    191933664: ["OOE"],
    191965113: ["OOE"],
    196433833: ["OOE"],
    196740034: ["OOE"],
    196920546: ["OOE"],
    197587400: ["QLP"],
    197755658: [16],
    198011271: ["OOE"],
    198076334: ["OOE"],
    198537349: ["OOE"],
    199574208: ["QLP"],
    199708835: ["OOE"],
    199709055: ["OOE"],
    201256771: ["OOE", 28],
    201924791: ["OOE", "QLP"],
    202445608: ["OOE"],
    203001339: ["OOE"],
    207080350: ["OOE"],
    215403644: ["QLP"],
    219222201: ["QLP"],
    219238496: ["QLP"],
    219435542: ["QLP"],
    220042007: [6, 32],
    228760807: ["OOE"],
    230065962: ["OOE"],
    231293332: ["OOE"],
    231822771: ["QLP", "OOE"],
    231851620: ["QLP"],
    233531814: ["OOE"],
    233650330: ["OOE"],
    234290005: ["OOE"],
    234347951: ["QLP"],
    242799145: ["OOE"],
    243662768: ["OOE"],
    245863693: ["OOE"],
    250195155: ["OOE"],
    251087387: ["QLP"],
    255908136: ["OOE"],
    255921197: [86],
    257737073: ["OOE"],
    260161144: [38, 39, 61, 62, 63, 64, 65, 66, 67, 68, 69, 87, 88],
    260502102: [65],
    261656371: ["QLP"],
    262958558: ["QLP"],
    266920154: ["OOE"],
    270648838: [88, "OOE"],
    271040947: ["OOE"],
    271529021: ["QLP"],
    271773721: ["OOE"],
    271924412: ["QLP"],
    272369124: ["OOE"],
    273566363: [76],
    273697554: ["QLP"],
    273792220: ["OOE"],
    274039311: ["OOE"],
    274347939: ["OOE"],
    274575202: ["OOE"],
    276935987: ["QLP"],
    278706358: ["OOE"],
    278988794: [5],
    279537689: ["OOE"],
    279741942: ["OOE"],
    279871974: ["OOE"],
    280909668: ["QLP"],
    281494100: ["OOE"],
    284613090: ["OOE"],
    286191384: ["OOE"],
    289044378: ["OOE"],
    290476605: ["OOE"],
    292013448: ["OOE"],
    291466214: ["OOE"],
}


class TICExcluded(set):
    """Record changes to the set of excluded data in the database."""

    _exclude_names = {0: "BLSOOE", -1: "OOE", -2: "QLP", -3: "SPOC"}
    _exclude_flags = {name: flag for flag, name in _exclude_names.items()}

    def _update_db(self):
        """Make sure the database matches the given exclusion names."""

        in_db = exclude_data[self.tic_id]
        with CacheSession.begin() as cache:  # pylint: disable=no-member
            for drop in in_db - self:
                cache.execute(
                    delete(ExcludeDataTable).filter_by(
                        tic_id=self.tic_id,
                        exclude=self._exclude_flags.get(drop, drop),
                    )
                )

            for add in self - in_db:
                cache.add(
                    ExcludeDataTable(
                        tic_id=self.tic_id,
                        exclude=self._exclude_flags.get(add, add),
                    )
                )

    def __init__(self, tic_id, exclude_flags):
        """Remember TIC ID associated with exclusions."""

        self.tic_id = tic_id
        super().__init__(
            self._exclude_names[exclude] if exclude <= 0 else exclude
            for exclude in exclude_flags
        )

    def update_exclusions(self, method_name, *args, **kwargs):
        """Apply a set method and record the result in the database."""

        print(
            f"Applying {method_name} with self: {self!r}, args: {args!r}, "
            f"and kwargs: {kwargs!r}"
        )
        result = getattr(super(), method_name)(*args, **kwargs)
        self._update_db()
        return result

    __iand__ = partialmethod(update_exclusions, "__iand__")
    __ior__ = partialmethod(update_exclusions, "__ior__")
    __isub__ = partialmethod(update_exclusions, "__isub_")
    __ixor__ = partialmethod(update_exclusions, "__ixor__")
    __rand__ = partialmethod(update_exclusions, "__rand__")
    __ror__ = partialmethod(update_exclusions, "__ror__")
    __rsub__ = partialmethod(update_exclusions, "__rsub__")
    __rxor__ = partialmethod(update_exclusions, "__rxor__")
    add = partialmethod(update_exclusions, "add")
    clear = partialmethod(update_exclusions, "clear")
    difference_update = partialmethod(update_exclusions, "difference_update")
    discard = partialmethod(update_exclusions, "discard")
    intersection_update = partialmethod(
        update_exclusions, "intersection_update"
    )
    pop = partialmethod(update_exclusions, "pop")
    remove = partialmethod(update_exclusions, "remove")
    symmetric_difference_update = partialmethod(
        update_exclusions, "symmetric_difference_update"
    )
    update = partialmethod(update_exclusions, "update")


class ExcludeData:  # pylint: disable=too-few-public-methods
    """Mapping from TIC ID to what to exclude from modeling."""

    def __getitem__(self, tic_id):
        """Query the exclusions for the given TIC ID."""

        with CacheSession.begin() as cache:  # pylint: disable=no-member
            return TICExcluded(
                tic_id,
                cache.scalars(
                    select(ExcludeDataTable.exclude).filter_by(tic_id=tic_id)
                ).all(),
            )

    def get(self, tic_id, default):
        """Get the item or default if it does not exist."""

        with CacheSession.begin() as cache:  # pylint: disable=no-member
            exclude_flags = cache.scalars(
                select(ExcludeDataTable.exclude).filter_by(tic_id=tic_id)
            ).all()
            if not exclude_flags:
                return default
            return TICExcluded(tic_id, exclude_flags)


exclude_data = ExcludeData()


def add_manual_exclusions():
    """Add the menual exclusions to the database."""

    for tic_id, exclude_list in _manual_exclude_data.items():
        already_excluded = exclude_data[tic_id]
        already_excluded |= set(exclude_list)


if __name__ == "__main__":
    add_manual_exclusions()
